import os
import random
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor
from datasets import Dataset
from sklearn.model_selection import KFold
from peft import LoraConfig, get_peft_model
from charsiu.model.aligner import CharsiuAligner
from charsiu.preprocess import load_audio, extract_features

# Set random seed
random.seed(1234)
torch.manual_seed(1234)

# Load participant sets CSV
participant_df = pd.read_csv("participant_sets.csv")
training_ids = participant_df[participant_df['set'].str.lower() == 'training']['ParticipantID'].str.lower().tolist()


# Extract info from file names
base_dir = "KT1"
data = []

# "K1[ppt_id]6participant_word.wav"
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.wav') and "participant_" in file:
                base = file[:-4]  # remove '.wav'
                try:
                    # Split at 'participant_'
                    before, word = base.split("participant_")
                    # Remove 'K1' prefix and trailing digit before 'participant_'
                    ppt_id = before.replace("K1", "")[:-1]
                    if ppt_id.lower() in training_ids:
                        data.append({
                            "file_name": os.path.join(folder_path, file),
                            "ppt_id": ppt_id,
                            "transcription": word
                        })
                except ValueError:
                    # Skip if the split doesn't work
                    continue


df = pd.DataFrame(data)

# Load Charsiu
model = CharsiuAligner(model_name="charsiu/en_w2v2_fs_10ms")

# Configure LoRA layer
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Preprocessing function
def preprocess(batch):
    audio_tensor, sr = load_audio(batch["file_name"])
    features = extract_features(audio_tensor, sr)
    return {
        "input_values": features,
        "labels": batch["transcription"]
    }

# Convert from pandas df to Dataset (better for neural networks)
dataset = Dataset.from_pandas(df)
dataset = dataset.map(preprocess, remove_columns=["file_name", "ppt_id", "transcription"])

# Training
from charsiu.training import CharsiuTrainer, CharsiuTrainingArguments
training_args = CharsiuTrainingArguments(
    output_dir="./lora_charsiu_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=5e-4,  # can change
    weight_decay=0.01,
    num_train_epochs=1,
    fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=10,
)

# Cross-validation
kf = KFold(n_splits=8, shuffle=True, random_state=1234)
for epoch in range(10):
    print(f"Epoch {epoch + 1}/10")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"  Fold {fold + 1}/8")
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)

        trainer = CharsiuTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

# Save the final model
model.save_pretrained("./lora_charsiu_finetuned")
print("LoRA Fine-tuning Complete! Model saved to ./lora_charsiu_finetuned")

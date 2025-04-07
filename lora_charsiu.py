import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load Charsiu 
model_name = "charsiu/en_w2v2_fc_10ms"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Apply LoRA Adaptation
lora_config = LoraConfig(
    r=8,  # Rank; experiment with different numbers
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,  # Regularization
    bias="none",
    target_modules=["q_proj", "v_proj"],  # Modify attention layers
)

# Apply LoRA modifications
lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# Load and Preprocess Dataset
dataset_path = "path/to/your/dataset"  # Change this to your dataset path
dataset = load_dataset(dataset_path)

def preprocess(batch):
    """Process audio and text into model-compatible format"""
    audio = batch["audio"]
    transcription = batch["text"]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs["labels"] = processor.tokenizer(transcription, return_tensors="pt")["input_ids"]
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess, remove_columns=["audio", "text"])

# Define Training Configuration
training_args = TrainingArguments(
    output_dir="./lora_charsiu_finetuned",
    per_device_train_batch_size=2,  # Reduce if memory issues
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-4,
    weight_decay=0.01,
    num_train_epochs=3,
    fp16=True,  # Enables mixed precision training
    gradient_accumulation_steps=8,  # Helps with low-memory GPUs
    push_to_hub=False,
)

### Step 5: Train the Model
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

### Step 6: Save Fine-Tuned Model
lora_model.save_pretrained("./lora_charsiu_finetuned")
print("✅ LoRA Fine-tuning Complete! Model saved to ./lora_charsiu_finetuned")

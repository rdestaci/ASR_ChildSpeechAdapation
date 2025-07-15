# Adaptation of forced alignment ASR systems to child speech

## Project Overview

This project aims to improve transcription accuracy of automatic speech recognition (ASR) and forced alignment systems (specifically Montreal Forced Aligner) to child speech, and examine the overall efficiency of using these tools for research purposes.

This a graduate capstone project for the Compuational Social Science M.S. program.

## Methods

### Data

For this evaluation of ASR and forced alignment models, we adapt pre-trained models using 72-participant child speech dataset.

This dataset consists of speech recordings of 24 single-word utterances, collected through the administration of the Goldman-Fristoe Test of Articulation (GFTA) on male and female child speakers ranging from ages 3 to 5 years old. In accordance with Insitutional Review Board approval, this dataset is not available for public use.

### ASR: Evaluating methods to fine-tune Wav2Vec2 for ASR

To address relatively poor performance of Wav2Vec2 ASR on specifically child-produced single word utterances, we fine-tune Wav2Vec2 using the data.

### Forced Alignment: Adapting Montreal Forced Aligner to our data

To adapt Montreal Forced Aligner, we use the model's `mfa adapt` workflow. More details of this functionality can be found on [Montreal Forced Aligner user guide](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/adapt_acoustic_model.html). To optimize this adaptation for child speech, we build a custom dictionary mapping common variabilities of phone realizations to canonical transcriptions. 

## Results and Evaluation

Fine-tuning the wav2vec2-base-960h ASR model to improve performance on child speech proved challenging, largely due to the limited size of the training dataset and the need for more extensive training. Despite these efforts, a substantial performance gap between child and adult speech remained, indicating that significantly more data and training time are required to achieve more robust performance for child speech. Susceptibility to frequency biases in the training data—-driven by the overrepresentation of certain phones or phonological patterns—-affects predictions across both age groups, pointing to a broader constraint of the model rather than an age-related shortcoming.

Forced alignment efforts resulted in similar shortcomings. While dictionary adaptation shows some promise for future work, neither acoustic model adaptation nor dictionary adaptation resulted in marked improvement over the default acoustic model or dictionary for forced alignment of child speech.

The full analysis and discussion of results are available in `final_report.pdf`.

## License

[MIT License](https://opensource.org/license/mit/)

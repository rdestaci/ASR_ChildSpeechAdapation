# Adaptation of forced alignment ASR systems to child speech

## Project Overview

This project aims to improve transcription accuracy of automatic speech recognition forced alignment systems (specifically Montreal Forced Aligner and Charsiu) to child speech, and examine the overall efficiency of using these tools for research purposes.

This a graduate capstone project for the Compuational Social Science M.S. program.

## Methods

### Data

We adapt both Montreal Forced Aligner and Charsiu to a 72-participant child speech dataset.

This dataset consists of speech recordings of 24 single-word utterances, collected through the administration of the Goldman-Fristoe Test of Articulation (GFTA) on male and female child speakers ranging from ages 3 to 5 years old.

# Model Adaptation

To adapt Charsiu, we built a custom Python pipeline to employ a Low Rank Adaptation (LoRA) to the pre-trained attention layers of the existing neural network system, trained on our dataset.

To adapt Montreal Forced Aligner, we use the model's `mfa adapt` workflow. More details of this functionality can be found on [Montreal Forced Aligner user guide](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/adapt_acoustic_model.html).

## Results and Evaluation

## Further Directions

## License

[MIT License](https://opensource.org/license/mit/)
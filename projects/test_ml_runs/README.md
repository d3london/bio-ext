# BRCA Classifier

## Objectives
- [ ] Download pre-trained BERT type model from HuggingFace
- [ ] Train / test / valid split on set of .jsons
- [ ] Set up training parameters
- [ ] MLflow Experiment tracking
- [ ] log fine tuned model (e.g. using mlflow.transformers.log_model)

## Questions

1. the pre-trained bert model weights usually is saved to root ~/.cache is this ok?

2. and where do we want to save the trained model?
3. The trained model will be deployed and the documents labelled from the model will be human reviewed. Correct? Therefore, we want a model that is optimised for high recall (i.e., more sensitive)
- we should prioritise for recall over accuracy or F1 score

## Task
Text classification task
Search terms used in hugging face
"medical" , "bio", "biomarker", "clinical", "oncology"

Candidate 1: https://huggingface.co/bvanaken/clinical-assertion-negation-bert
Candidate 2: https://huggingface.co/michiyasunaga/BioLinkBERT-large
Candidate 3: https://huggingface.co/blizrys/biobert-v1.1-finetuned-pubmedqa

Candidate 2 chosen as large volume of downloads and also use case looks similar to ours.


1. Data generated from doccano lives in `data/admin.csv`

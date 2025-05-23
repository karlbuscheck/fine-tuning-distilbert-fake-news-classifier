# Fine-tuning DistilBERT to Build a Fake News Classifier

In this project, we’ll build a binary classification model to detect fake news using **DistilBERT**, a streamlined version of **BERT** — one of the foundational large language models (LLMs). We’ll fine-tune a pretrained DistilBERT model on our fake news dataset, evaluate how well it generalizes to *unseen* news articles, and then test it on a very *real* story from the front page of *The New York Times*.

You can check out baseline model we'll be using on Hugging Face [right here](https://huggingface.co/distilbert/distilbert-base-uncased). 

View the full notebook right here:

[![Open In Colab]((https://colab.research.google.com/drive/17SPfMHY63F5r3SOxO1w0foKuHGOkoZpu?usp=sharing))

## The Pipeline

Before we dive in, here's a quick rundown of what is to come:

- Setup and data import
- Load the pretrained model from Hugging Face
- Prepare the dataset for PyTorch
- Fine-tune the model
- Generate predictions on unseen (test) data
- Measure classification performance
- Save the model for future use
- Use fine-tuned DistilBERT to classify a *real* news story
- Build the `classify_article` function

## Tools & Libraries Used

- **Google Colab** — for free GPU access and seamless integration with Drive  
- **Python 3.11.12** — base language for all modeling and data wrangling  
- **pandas** — for reading and managing the dataset of 31K news articles
- **numpy** - for numerical operations and array manipulation
- **gdown:** - for programmatically downloading files from Google Drive within the Colab environment
- **transformers** — Hugging Face's library for working with DistilBERT and custom fine-tuning  
- **torch** — the underlying deep learning framework for model fine-tuning and inference
- **scikit-learn** - for test-train split, classification metrics, and model evaluation   
- **shutil** — for saving and exporting the fine-tuned model    
- **Google Drive** — used to store and export the final model

## Acknowledgements

This project started with a notebook I was working on for my NLP class, taught by  Associate Professor Michele Samorani at Santa Clara University. Thanks to the professor and the Leavey School of Business for providing the scaffolding on which this DistilBERT fake news classifier was built.

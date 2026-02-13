# Fine-tuning DistilBERT to Build a Fake News Classifier

In this project, we’ll build a binary classification model to detect fake news using **DistilBERT**, a streamlined version of **BERT** — one of the foundational large language models (LLMs). We’ll fine-tune a pretrained DistilBERT model on our fake news dataset, evaluate how well it generalizes to *unseen* news articles, and then test it on a very *real* story from the front page of *The New York Times*.

You can check out the baseline model we'll be using on Hugging Face [right here](https://huggingface.co/distilbert/distilbert-base-uncased). 

## View the full notebook right here:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17SPfMHY63F5r3SOxO1w0foKuHGOkoZpu?usp=sharing)

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

## ⚠️ Evaluation Notes & Known Limitations

During a post-hoc audit of the dataset and evaluation pipeline, I identified a couple of important considerations regarding model performance and generalization.

**Duplicate Data & Potential Leakage**

The dataset contains a non-trivial number of duplicate articles (exact duplicate content strings). ~5,793 duplicate rows were identified in the full dataset. In a random train/test split, ~1,533 identical articles appeared in both the training and test sets. If duplicates are not removed before splitting, this creates data leakage, allowing the model to effectively “memorize” examples that later appear in the test set. This can inflate reported accuracy.

Recommended Fix

Deduplicate before splitting:

`fake_news = fake_news.drop_duplicates(subset="content").reset_index(drop=True)`

Then re-run the train/test split and training process.
Note: Reported test accuracy may decrease after deduplication, but will better reflect true generalization.

**Reproducibility**

Fine-tuning transformer models involves stochastic processes (random initialization, data shuffling, dropout). Without setting a random seed, confidence scores and even metrics may vary slightly between runs.

For more reproducible results:

```
import random, numpy as np, torch
from transformers import set_seed

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
set_seed(seed)
```
## Acknowledgements

This project started with a notebook I was working on for my NLP class, taught by  Associate Professor Michele Samorani at Santa Clara University. Thanks to the professor and the Leavey School of Business for providing the scaffolding on which this DistilBERT fake news classifier was built.

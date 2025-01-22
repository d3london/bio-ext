# CODE NEEDS REFACOTIRNG AND TIDY
# this script is adapted from wine_test.py
# will also work against objectives outlined in directory readme.md

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# RANDOM NEED TO SAVE THIS IN CONFIG 
RANDN = 45

# Prepare data for transformers ready 
df = pd.read_csv("data/admin.csv")


def train_test_split(data,ratio,xcol,targetcol,randomstate = RANDN):
    """Custom train test split capturing random state, 
    after shuffling of pandas dataframe

    Args:
        data (pandas dataframe): dataframe to be shuffled and split
        ratio (num): number [0,1] to allow train test split
        xcol(str): pandas column for training
        targetcol(str):pandas column name for labels
        randomstate (num, optional): random state . Defaults to RANDN.

    Returns:
        train,test: train and test both returned 
    """
    train_ratio = ratio
    shuffled = data.sample(frac = 1, random_state=randomstate).reset_index(drop=True)
    shuffled = shuffled[[xcol,targetcol]]
    n_size = int(len(shuffled) * train_ratio)
    train = shuffled[:n_size]
    test = shuffled[n_size:]
    return train,test


dftrain,dftest = train_test_split(data=df,ratio = 0.8,xcol = "text",targetcol="label")
dataset = Dataset.from_pandas(dftrain)
# 
def gen_tokens(data,text,tokenizer = tokenizer):
    return tokenizer(data[text],padding = True, truncation = True, return_tensors="pt")

# Tokenizer(first run time 3 minutes)
tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-large')
model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-large')


inputs = gen_tokens(data= dataset, text = "text",tokenizer = tokenizer)
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

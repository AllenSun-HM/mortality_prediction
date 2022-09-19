import json

import numpy as np
import pandas as pd

train_demo_df = pd.read_csv("embeddings/train_demo_embedding.csv")
val_demo_df = pd.read_csv("embeddings/val_demo_embedding.csv")
train_ts_df = pd.read_csv("embeddings/ts_train_embedding.csv")
val_ts_df = pd.read_csv("embeddings/ts_val_embedding.csv")
train_text_df = pd.read_csv("embeddings/train_text_embedding.csv")
val_text_df = pd.read_csv("embeddings/val_text_embedding.csv")

train_df = pd.merge(pd.merge(train_text_df, train_ts_df, on='hadm_id'), train_demo_df, on='hadm_id')
val_df = pd.merge(pd.merge(val_text_df, val_ts_df, on='hadm_id'), val_demo_df, on='hadm_id')

def convert(x, zero_filling_dim=None):
    if type(x) == float:
        return np.zeros(zero_filling_dim).tolist()
    x_arr = json.loads(x)
    return x_arr

train_df['text_embedding'] = train_df['text_embedding'].map(convert)
train_df['ts_embedding'] = train_df['ts_embedding'].map(convert)
train_df['demo_embedding'] = train_df['demo_embedding'].map(convert)
train_df['embedding'] = train_df['text_embedding'] + train_df['ts_embedding'] + train_df['demo_embedding']
train_df = train_df[['hadm_id', 'mortality', 'embedding']]

val_df['text_embedding'] = val_df['text_embedding'].map(lambda x: convert(x, 768))
val_df['ts_embedding'] = val_df['ts_embedding'].map(convert)
val_df['demo_embedding'] = val_df['demo_embedding'].map(convert)
val_df['embedding'] = val_df['text_embedding'] + val_df['ts_embedding'] + val_df['demo_embedding']
val_df = val_df[['hadm_id', 'mortality', 'embedding']]

train_df.to_csv('embeddings/all_train.csv', index=False)
val_df.to_csv('embeddings/all_val.csv', index=False)

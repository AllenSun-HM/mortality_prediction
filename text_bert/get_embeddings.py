import math

import numpy
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm


'''
Get embeddings using the BioBERT model (paper of the model: https://arxiv.org/pdf/1901.08746.pdf) 
'''

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1")
model = AutoModel.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1").to(device)


train_id_df = pd.read_csv('../data/in-hospital-mortality/train_listfile.csv')
val_id_df = pd.read_csv('../data/in-hospital-mortality/val_listfile.csv')

mp_train_df = pd.read_csv('../data/clinical_notes/MP_IN_adm_train.csv')
mp_val_df = pd.read_csv('../data/clinical_notes/MP_IN_adm_val.csv')
mp_train_df = mp_train_df[mp_train_df['id'].isin(set(train_id_df['HADM_ID']))]
mp_val_df = mp_val_df[mp_val_df['id'].isin(set(val_id_df['HADM_ID']))]

# get training data embeddings
train_result_df = pd.DataFrame(columns=['hadm_id', 'text_embedding', 'mortality'])
train_result_df['text_embedding'] = train_result_df['text_embedding'].astype(object)
for i in tqdm(range(mp_train_df.shape[0])):
    id, text, hospital_expire_flag = mp_train_df.iloc[i]
    if isinstance(text, float) and math.isnan(text): # text is missing
        new_row = pd.DataFrame({'hadm_id': [id], 'text_embedding': [None], 'mortality': [hospital_expire_flag]},
                               columns=['hadm_id', 'text_embedding', 'mortality'])
        train_result_df = pd.concat([train_result_df, new_row], axis=0, ignore_index=True)
        continue
    embedding = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    embedding = model(**embedding)['pooler_output'].detach().numpy()[0]
    embedding = numpy.array(embedding).tolist()
    new_row = pd.DataFrame(
        {'hadm_id': [id], 'text_embedding': [embedding], 'mortality': [hospital_expire_flag]},
        columns=['hadm_id', 'text_embedding', 'mortality'])
    train_result_df = pd.concat([train_result_df, new_row], axis=0, ignore_index=True)
train_result_df.to_csv('../final_classifier/embeddings/train_text_embedding.csv', index=False)


# get validation data embeddings
val_result_df = pd.DataFrame(columns=['hadm_id', 'text_embedding', 'mortality'])
val_result_df['text_embedding'] = val_result_df['text_embedding'].astype(object)
for i in tqdm(range(mp_val_df.shape[0])):
    id, text, hospital_expire_flag = mp_val_df.iloc[i]
    if isinstance(text, float) and math.isnan(text): # if text is missing
        new_row = pd.DataFrame({'hadm_id': [id], 'text_embedding': [None], 'mortality': [hospital_expire_flag]},
                               columns=['hadm_id', 'text_embedding', 'mortality'])
        val_result_df = pd.concat([val_result_df, new_row], axis=0, ignore_index=True)
        continue
    embedding = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    embedding = model(**embedding)['pooler_output'].detach().numpy()[0]
    embedding = numpy.array(embedding).tolist()
    new_row = pd.DataFrame(
        {'hadm_id': [id], 'text_embedding': [embedding], 'mortality': [hospital_expire_flag]},
        columns=['hadm_id', 'text_embedding', 'mortality'])
    val_result_df = pd.concat([val_result_df, new_row], axis=0, ignore_index=True)
val_result_df.to_csv('../final_classifier/embeddings/val_text_embedding.csv', index=False)
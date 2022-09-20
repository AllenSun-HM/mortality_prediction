# mortality_prediction

## Step 1
populate the data folder with the output from [MIMICIII Preprocessing](https://github.com/AllenSun-HM/mimic3-data-preprocessing).

## Step 2 Train Modality-specific models and get embeddings
#### Demography MLP
Run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/demography_mlp/main.py) to get the model
Run [get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/demography_mlp/get_embeddings.py) to get the embeddings for demographic and diagnosis data

#### Clinical Notes(text) BERT
Run [get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/text_bert/get_embeddings.py) to get the embeddings for clinical notes


#### Timeseries Transformer
Run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/timeseries_transformer/main.py) with appropriate arguments to get the model. For example,
```{bash}
python3 src/main.py --output_dir experiments --comment "classification from Scratch" --name mimic_fromScratch --records_file Classification_records.xls --data_dir ~/data/in-hospital-mortality/train --data_class mimic --epochs 200 --lr 0.0001 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric AUROC --lr_step 80 --max_seq_len 48 
```
Run [ts_get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/timeseries_transformer/ts_get_embeddings.py) to get the embeddings for timeseries data. 

## Step 3 Merge embeddings and Train the final classifer
#### merge embeddings from all three modalities
run [merge_datasets.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/final_classifier/merge_datasets.py)
#### train the final classifer
run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/final_classifier/main.py) with arguments


More backgrounds are on this slides: https://docs.google.com/presentation/d/1goG7G8vwJqrdD4jlEP-fUQAl3T0gmCniBgmygE4wELY/edit#slide=id.gfdf24d3a7d_0_146

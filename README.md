# mortality_prediction

## Step 1
populate the [data](https://github.com/AllenSun-HM/mortality_prediction/data) folder with the output from [MIMICIII Preprocessing](https://github.com/AllenSun-HM/mimic3-data-preprocessing).

## Step 2 Train Modality-specific models and get embeddings
### MLP for demography & diagnosis data
#### 1. Run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/demography_mlp/main.py) to train the model.
```{bash}
python3 demography_mlp/main.py --lr 0.01 --n_epochs 100 --weight 1,6
```
You can tune the hyperparameter(lr, n_epochs, and weight) showed above to achieve the best result.

The format of weight is {the weight of no-mortality class},{the weight of has-mortality class}

The model will be saved at /demography_mlp/{model_id}_model.pth.

#### 2. Run [get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/demography_mlp/get_embeddings.py) to get the embeddings for demographic and diagnosis data. 

Embeddings will be stored at /final_classifier/embedding/train_demo_embedding.csv and /final_classifier/embeddings/val_demo_embedding.csv.

### BioBERT for clinical notes
#### 1. Run [get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/text_bert/get_embeddings.py).
This will input the clinical notes into the [BioBERT](https://arxiv.org/pdf/1901.08746.pdf) model and get the embeddings.

Embeddings will be stored at /final_classifier/embeddings/train_text_embedding.csv and /final_classifier/embeddings/val_text_embedding.csv.


### Transformer for timeseries data
#### 1. Run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/timeseries_transformer/main.py) with arguments to train the model.
For example,
```{bash}
python3 timeseries_transformer/main.py --output_dir experiments --name mimic_fromScratch --records_file Classification_records.xls --data_dir data/timeseries/train_val --data_class mimic --epochs 200 --lr 0.0001 --optimizer RAdam  --pos_encoding learnable  --task classification  --key_metric AUROC --lr_step 80 --max_seq_len 48 
```
* *output_dir*: the model will be saved at timeseries_transformer/output_dir

* *epochs*, *lr*, *lr_step*: tunable hyperparameters.

* *max_seq_len*: the number of data points considered by the model for each ICU stay. You should keep this to 48 as I only used the first 48 hours of each ICU stay with one data point per hour as the feature.
* *key_metric*: The metric to determine the best model. I used AUROC as this is an imbalanced dataset. 

More options and descriptions of these arguments can be found in [timeseries_transformer/options.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/timeseries_transformer/options.py).


#### 2. Run [ts_get_embeddings.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/timeseries_transformer/ts_get_embeddings.py) to get the embeddings for timeseries data. 
The embeddings will be stored at /final_classifier/embeddings/ts_train_embedding.csv and /final_classifier/embeddings/ts_val_embedding.csv.

## Step 3 Merge embeddings to train final classifier
### 1. merge embeddings from all three modalities
run [merge_datasets.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/final_classifier/merge_datasets.py) 

This will merge and store the merged embeddings at embeddings/all_train.csv and embeddings/all_val.csv.
### 2. train the final classifier
run [main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/final_classifier/main.py) with hyperparameter arguments.
```{bash}
python3 final_classifier/main.py --lr 0.001 --n_epochs 100 --weight 1,6
```
You can tune the hyperparameter(lr, n_epochs, and weight) above to achieve the best result.

[main.py](https://github.com/AllenSun-HM/mortality_prediction/blob/main/final_classifier/main.py) will print out the best AUROC achieved on the validation data.





#### More details including the model structure and benchmarking results are on this slides: https://docs.google.com/presentation/d/1goG7G8vwJqrdD4jlEP-fUQAl3T0gmCniBgmygE4wELY/edit#slide=id.gfdf24d3a7d_0_146

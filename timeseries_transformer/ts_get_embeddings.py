import pandas as pd
from torch.utils.data import DataLoader

from timeseries_transformer.datasets.dataset import ClassiregressionDataset, collate_superv
from timeseries_transformer.datasets.data import MimicData
import torch
from timeseries_transformer.models.ts_transformer import TSTransformerEncoderClassiregressor

if __name__ == '__main__':
    result_df = pd.DataFrame(columns=['hadm_id', 'ts_embedding'])
    result_df['ts_embedding'] = result_df['ts_embedding'].astype(object)

    my_data = MimicData("../data/in-hospital-mortality/train", n_proc=8)
    train_list_df = pd.read_csv('../data/in-hospital-mortality/train_listfile.csv')
    val_list_df = pd.read_csv('../data/in-hospital-mortality/val_listfile.csv')

    data = ClassiregressionDataset(my_data, my_data.all_IDs)
    model = TSTransformerEncoderClassiregressor(d_model=64, max_len=48, num_classes=2, num_layers=3, dropout=0.1, feat_dim=13, n_heads=8, dim_feedforward=256, output_embedding=True).to('cpu')
    loader = DataLoader(dataset=data,
                             batch_size=64,
                             shuffle=False,
                             pin_memory=True,
                             collate_fn=lambda x: collate_superv(x, max_len=model.max_len))
    # TODO: replace with the .pth file of the model that you want to use
    state_dict = torch.load("~/mvts_transformer/experiments/mimic_fromScratch_2022-08-10_10-38-13_odi/checkpoints/model_best.pth")
    model.load_state_dict(state_dict['state_dict'])
    model.eval()

    for i, batch in enumerate(loader):
        X, targets, padding_masks, IDs = batch
        ts_embedding = model(X, padding_masks).detach().numpy()
        new_row = pd.DataFrame(
            {'hadm_id': list(IDs), 'ts_embedding': ts_embedding.tolist()},
            columns=['hadm_id', 'ts_embedding'])
        result_df = pd.concat([result_df, new_row], axis=0, ignore_index=True)
    train_result_df = result_df[result_df['hadm_id'].isin(set(train_list_df['HADM_ID'].values))]
    val_result_df = result_df[result_df['hadm_id'].isin(set(val_list_df['HADM_ID'].values))]
    train_result_df.to_csv('../final_classifier/embeddings/ts_train_embedding.csv', index=False)
    val_result_df.to_csv('../final_classifier/embeddings/ts_val_embedding.csv', index=False)
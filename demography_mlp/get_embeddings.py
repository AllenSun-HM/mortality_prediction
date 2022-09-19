import numpy

from dataloader import train_loader, val_loader
import torch
from model import MimicDemographyModel
import pandas as pd

if __name__ == '__main__':
    train_result_df = pd.DataFrame(columns=['hadm_id', 'demo_embedding'])
    train_result_df['demo_embedding'] = train_result_df['demo_embedding'].astype(object)
    val_result_df = pd.DataFrame(columns=['hadm_id', 'demo_embedding'])
    val_result_df['demo_embedding'] = val_result_df['demo_embedding'].astype(object)

    model = MimicDemographyModel(output_embedding=True)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    for features, labels, IDs in train_loader:
        embedding = model(features).detach().numpy()
        embedding = numpy.array(embedding).tolist()
        new_row = pd.DataFrame(
            {'hadm_id': list(IDs.numpy()), 'demo_embedding': embedding},
            columns=['hadm_id', 'demo_embedding'])
        train_result_df = pd.concat([train_result_df, new_row], axis=0, ignore_index=True)
    train_result_df.to_csv('../final_classifier/embeddings/train_demo_embedding.csv', index=False)

    for features, labels, IDs in val_loader:
        embedding = model(features).detach().numpy()
        embedding = numpy.array(embedding).tolist()
        new_row = pd.DataFrame(
            {'hadm_id': list(IDs.numpy()), 'demo_embedding': embedding},
            columns=['hadm_id', 'demo_embedding'])
        val_result_df = pd.concat([val_result_df, new_row], axis=0, ignore_index=True)
    val_result_df.to_csv('../final_classifier/embeddings/val_demo_embedding.csv', index=False)
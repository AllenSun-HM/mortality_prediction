import random

import ray
import torch
from model import train, MimicDemographyModel
from dataloader import train_loader, val_loader, test_loader
from torch.utils.tensorboard import SummaryWriter
from ray import tune
from model import test


'''
grid search the best hyperparameters for the model
'''

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter()
    id = random.randint(1, 999999999)
    train(config, train_loader, val_loader, writer, id)
    model = MimicDemographyModel().to(device)
    model.load_state_dict(torch.load(str(id)+'_model.pth'))
    test_loss, test_acc, AUROC, AUPRC = test(test_loader, model, config)
    tune.report(accuracy=test_acc)
    tune.report(loss=test_loss)
    tune.report(auroc=AUROC)
    tune.report(auprc=AUPRC)

if __name__ == '__main__':
    ray.init()
    analysis = tune.run(main, resources_per_trial={'gpu': 1}, config={"lr": tune.grid_search([0.0001, 0.001, 0.01, 0.1]), 'weight': tune.grid_search([[1, 3], [1,4], [1,5], [1,6]]), 'n_epochs': 100})

print("Best config for acc: ", analysis.get_best_config(metric="accuracy", mode='max'))
print("Best config for loss: ", analysis.get_best_config(metric="loss", mode='min'))
print("Best config for auroc: ", analysis.get_best_config(metric="auroc", mode='max'))
print("Best config for auprc: ", analysis.get_best_config(metric="auprc", mode='max'))

print("Best accuracy: ", analysis.get_best_trial(metric="accuracy", mode='max'))

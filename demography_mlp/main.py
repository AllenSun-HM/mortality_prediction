import argparse
import random
import time

import torch
from model import train, MimicDemographyModel
from dataloader import train_loader, val_loader, test_loader
from torch.utils.tensorboard import SummaryWriter
from model import test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr")
    parser.add_argument("--n_epochs")
    parser.add_argument("--weight")
    args = parser.parse_args()
    weights = args.weight.split(',')
    config = {}
    config['weight'] = [float(weight) for weight in weights]
    config['n_epochs'] = int(args.n_epochs)
    config['lr'] = float(args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter()
    model_id = time.time()
    train(config, train_loader, val_loader, writer, model_id) # train the model and save the model at {model_id}_model.pth
    model = MimicDemographyModel().to(device)
    model.load_state_dict(torch.load(str(model_id)+"_model.pth"))
    test_loss, test_acc, AUROC, AUPRC = test(test_loader, model, config)
    print({'test_loss': test_loss, 'test_acc': test_acc, 'AUROC': AUROC, 'AUPRC': AUPRC})

if __name__ == '__main__':
    main()
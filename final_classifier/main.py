import argparse
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from model import EmbeddingClassifier, train
from dataloader import train_loader, val_loader

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
    model = EmbeddingClassifier()

    writer = SummaryWriter()

    id = time.time()
    best_val_auroc = train(model, device, config, train_loader, val_loader, writer, id)
    print(f"best val auroc:{best_val_auroc:.6f}")


if __name__ == '__main__':
    main()

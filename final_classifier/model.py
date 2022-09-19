import torch.nn as nn
import torch
import torch.optim as optim

from metrics import cross_entropy_loss, accuracy, auroc, auprc


class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingClassifier, self).__init__()
        self.module1 = nn.Linear(3904, 100)
        self.module2 = nn.Linear(100, 100)
        self.module3 = nn.Linear(100, 64)
        self.output = nn.Linear(64, 2)


    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.output(x)
        return x

def train(model, device, config, train_dataloader, val_dataloader, writer, id):
    max_auroc = 0
    num_batches = len(train_dataloader)
    n_epochs = config['n_epochs']
    lr = config['lr']
    weights = config['weight']
    weights = torch.tensor(weights, dtype=torch.float, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train() # set to train
    for i in range(n_epochs):
        train_loss = 0
        train_acc = 0
        auroc_score = 0
        auprc_score = 0

        print(f"Epoch:{i+1}")
        for batch_idx, (inputs, labels, ids) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = cross_entropy_loss(outputs, labels, weights)
            predictions = torch.argmax(outputs, axis=1)
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(outputs).float()
            auroc_score += auroc(probs.cpu().numpy(), labels.cpu().numpy())
            auprc_score += auprc(probs.cpu().numpy(), labels.cpu().numpy())
            acc = accuracy(predictions, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"current batch:{batch_idx}/all:{num_batches}, loss:{loss.item():.6f}")

            train_loss += loss.item()
            train_acc += acc
        train_loss = train_loss / num_batches
        train_acc = train_acc / num_batches
        auroc_score = auroc_score / len(train_dataloader)
        auprc_score = auprc_score / len(train_dataloader)
        writer.add_scalar("loss/train", float(train_loss), i)
        writer.add_scalar("Accuracy/train", train_acc, i)
        writer.add_scalar("auroc/train", auroc_score, i)
        writer.add_scalar("auprc/train", auprc_score, i)
        print(f"average training loss:{train_loss:.6f}")
        print(f"average training accuracy:{train_acc:.6f}")
        print(f"average training auroc:{auroc_score:.6f}")
        print(f"average training auprc:{auprc_score:.6f}")

        def val(dataloader, model, criterion):
            val_loss = 0
            val_acc = 0
            auroc_score = 0
            auprc_score = 0

            val_num_batches = len(dataloader)

            model.eval()
            with torch.no_grad():
                for inputs, labels, ids in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)  # GPU
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels, weights)
                    probs = softmax(outputs).float()
                    auroc_score += auroc(probs.cpu().numpy(), labels.cpu().numpy())
                    auprc_score += auprc(probs.cpu().numpy(), labels.cpu().numpy())
                    predictions = torch.argmax(outputs, axis=1)
                    val_acc += accuracy(predictions, labels)
            val_loss = val_loss / val_num_batches
            val_acc = val_acc / val_num_batches
            auroc_score = auroc_score / len(dataloader)
            auprc_score = auprc_score / len(dataloader)
            return val_loss, val_acc, auroc_score, auprc_score


        #validation
        val_loss, val_acc, val_auroc, val_auprc = val(val_dataloader, model, cross_entropy_loss)
        writer.add_scalar("loss/val", float(val_loss), i)
        writer.add_scalar("Accuracy/val", val_acc, i)
        writer.add_scalar("auroc/val", val_auroc, i)
        writer.add_scalar("auprc/val", val_auprc, i)

        print(f"average val loss:{val_loss:.6f}")
        print(f"average val auroc:{val_auroc:.6f}")
        print(f"average val auprc:{val_auprc:.6f}")

        if val_auroc > max_auroc:
            max_auroc = val_auroc
            path = str(id) + '_model.pth'
            torch.save(model.state_dict(), path)
    writer.close()
    return max_auroc

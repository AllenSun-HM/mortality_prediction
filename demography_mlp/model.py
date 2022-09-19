import torch.nn as nn
import torch
import torch.optim as optim
from metrics import accuracy
from metrics import cross_entropy_loss
from metrics import auroc, auprc

class MimicDemographyModel(nn.Module):
    def __init__(self, output_embedding=False):
        super(MimicDemographyModel, self).__init__()
        self.activation = nn.Sigmoid()
        self.module1 = nn.Linear(29, 100)
        self.norm1 = nn.BatchNorm1d(100)
        self.module2 = nn.Linear(100, 100)
        self.norm2 = nn.BatchNorm1d(100)
        self.module3 = nn.Linear(100, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(64, 2)
        self.output_embedding = output_embedding

    def forward(self, x):
        x = self.module1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.module2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.module3(x)
        if not self.output_embedding:
            x = self.dropout(x)
            x = self.output(x)
        return x

device="cuda" if torch.cuda.is_available() else "cpu"

model = MimicDemographyModel().to(device)

def train(config, train_dataloader, val_dataloader, writer, id):
    min_val_loss = 1000
    num_batches = len(train_dataloader)
    n_epochs = config['n_epochs']
    lr = config['lr']
    weights = config['weight']
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train() # set to train
    for i in range(n_epochs):
        train_loss = 0
        train_acc = 0
        print(f"Epoch:{i+1}")
        for batch_idx, (inputs, labels, ids) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cross_entropy_loss(outputs, labels, weights)
            predictions = torch.argmax(outputs, axis=1)
            acc = accuracy(predictions, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"current batch:{batch_idx}/all:{num_batches}, loss:{loss.item():.6f}")

            train_loss += loss.item()
            train_acc += acc
        train_loss = train_loss / num_batches
        train_acc = train_acc / num_batches
        writer.add_scalar("loss/train", float(train_loss), i)
        writer.add_scalar("Accuracy/train", train_acc, i)
        print(f"average training loss:{train_loss:.6f}")

        def val(dataloader, model, criterion):
            val_loss = 0
            val_acc = 0
            val_num_batches = len(dataloader)

            model.eval()
            with torch.no_grad():
                for inputs, labels, ids in dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)  # GPU
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels, weights)
                    predictions = torch.argmax(outputs, axis=1)
                    val_acc += accuracy(predictions, labels)
            val_loss = val_loss / val_num_batches
            val_acc = val_acc / val_num_batches
            return val_loss, val_acc


        #validation
        val_loss, val_acc = val(val_dataloader, model, cross_entropy_loss)
        writer.add_scalar("loss/val", float(val_loss), i)
        writer.add_scalar("Accuracy/val", val_acc, i)
        print(f"average val loss:{val_loss:.6f}")
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            path = str(id) + '_model.pth'
            torch.save(model.state_dict(), path)
    writer.close()



def test(dataloader, model, config):
    weights = config['weight']
    test_loss = 0
    test_acc = 0
    test_num_batches = len(dataloader)

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            print(inputs.shape)
            inputs, labels = inputs.to(device), labels.to(device)  # GPU
            outputs = model(inputs)
            test_loss += cross_entropy_loss(outputs, labels, weights)
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(outputs).float()
            AUROC = auroc(probs.cpu().numpy(), labels.cpu().numpy())
            AUPRC = auprc(probs.cpu().numpy(), labels.cpu().numpy())
            predictions = torch.argmax(probs, axis=1)
            test_acc += accuracy(predictions, labels)
    test_loss = test_loss / test_num_batches
    test_acc = test_acc / test_num_batches
    return test_loss.item(), test_acc, AUROC, AUPRC
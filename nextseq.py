from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from data import NextSeq7Dataset
from torch.utils.data.dataloader import DataLoader


tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
model = AutoModel.from_pretrained("ethanyt/guwenbert-base")


class FineTune(nn.Module):
    def __init__(self,model):
        super(FineTune, self).__init__()
        self.model = model
        self.fc1 = nn.Linear(768, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = model(x).last_hidden_state
        output = output[:, 0, :]
        x = self.fc1(output)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze()


def evaluate_accuracy(dataloader,net):
    correct = 0
    n = 0
    net.eval()
    for i, data in enumerate(dataloader):
        sequence, labels = data
        #sequence.squeeze_()
        out = net(sequence)
        correct += (out.ge(0.5) == labels).sum().item()
        n += labels.shape[0]
    return correct / n

def fine_tune():
    train_data = NextSeq7Dataset('dataset/nextseq_poems_7_train.txt',tokenizer)
    dev_data = NextSeq7Dataset('dataset/nextseq_poems_7_dev.txt',tokenizer)
    test_data = NextSeq7Dataset('dataset/nextseq_poems_7_test.txt',tokenizer)
    train_loader = DataLoader(dataset=train_data,
                               batch_size=2048,
                               shuffle=True)
    dev_loader = DataLoader(dataset=dev_data,
                              batch_size=8192,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_data,
                            batch_size=8192,
                            shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    finetune = FineTune(model).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, finetune.parameters()), lr=0.0001)

    for epoch in range(10):
        train_loss = 0
        for i, data in enumerate(train_loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            finetune.train()
            sequence, labels = data
            labels = labels.float()
            out = finetune(sequence)
            l = loss(out, labels)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item()
            #print(train_loss)
        dev_acc = evaluate_accuracy(dev_loader, finetune)
        print('epoch %d ,loss %.4f' % (epoch + 1, train_loss) + ', dev acc {:.2f}%'.format(dev_acc * 100))
    test_acc = evaluate_accuracy(test_loader, finetune)
    print('test acc {:.2f}%'.format(test_acc * 100))
    torch.save(finetune.model,'./model/fine_tune_model.bin')

if __name__ == '__main__':
    fine_tune()
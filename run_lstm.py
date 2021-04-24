import torch
import argparse
from data import LSTMDataset
from utils import ratio_split
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
# parser.parse_args()
args = parser.parse_args()
args.dataset_path = 'data/replace_poems_7.txt'


class EncoderRNNWithVector(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, n_layers=1, batch_size=4):
        super(EncoderRNNWithVector, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.out_size = out_size
        
        # 输入层转换
        self.in_ = torch.nn.Linear(input_size, hidden_size)
        
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

        # 加了一个线性层，全连接
        self.out = torch.nn.Linear(hidden_size, out_size)
        
    def forward(self, word_inputs, hidden):

        # batch, time_seq, input
        inputs = self.in_(word_inputs)

        output, hidden = self.gru(inputs, hidden)

        output = self.out(output)

        # the last of time_seq
        output = output[:,-1,:]

        return output, hidden

    def init_hidden(self):

        hidden = torch.autograd.Variable(torch.zeros(self.n_layers, 4, self.hidden_size), r)
        return hidden

    
if __name__ == '__main__':

    dataset = LSTMDataset(args.dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    model = EncoderRNNWithVector(6361, 200, 16)
    train = DataLoader(train_dataset, batch_size=4, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    encoder_hidden = model.init_hidden()

    for (i, data) in enumerate(train):

        inputs, label = data
        # inputs = inputs.cuda(n)
        encoder_outputs, encoder_hidden = model(inputs, encoder_hidden)

        optimizer.zero_grad()
        loss = criterion(encoder_outputs, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        print("loss: ", loss.data.item())


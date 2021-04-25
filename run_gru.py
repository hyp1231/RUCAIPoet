import torch
import argparse
from data import LSTMDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.device = 0
args.epochs = 100
args.lr = 3e-4
args.decay = 0.
args.batch_size = 256
args.num_workers = 0
args.dataset_path = 'dataset/rpl_all/replace_poems_all'

args.word_num = 6357
args.hd_size = 200
args.pos = 16
args.wd_ans = 6357
args.alpha = 1.


class EncoderRNNWithVector(torch.nn.Module):
    def __init__(self, input_size, hidden_size, pos_size, wd_size, n_layers=1):
        super(EncoderRNNWithVector, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.pos_size = pos_size
        self.wd_size = wd_size

        # 输入层转换
        self.in_ = torch.nn.Embedding(input_size, hidden_size)

        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)

        # 加了一个线性层，全连接
        self.pos_out = torch.nn.Linear(hidden_size, pos_size)
        self.wd_out = torch.nn.Linear(hidden_size, wd_size)

    def forward(self, word_inputs):

        # batch, time_seq, input
        inputs = self.in_(word_inputs)

        output, _ = self.gru(inputs)
        output = output[:,-1,:]

        pos_output = self.pos_out(output)
        wd_output = self.wd_out(output)

        return pos_output, wd_output

def eval_(eval_data, model):
    for i, (inputs, pos_label, wd_label) in tqdm(enumerate(train_data)):
        inputs = inputs.to(device)
        pos_label = pos_label.to(device)
        wd_label = wd_label.to(device)
        pos_output, wd_output = model(inputs)
        _, max_pos_idx = torch.topk(pos_output, k=1, dim=-1)
        _, max_wd_idx = torch.topk(wd_output, k=1, dim=-1)
        pos_acc = (max_pos_idx.squeeze(-1) == pos_label).sum() / pos_label.shape[0]
        wd_acc = (max_wd_idx.squeeze(-1) == wd_label).sum() / wd_label.shape[0]
        tot_acc = torch.dot((max_pos_idx.squeeze(-1) == pos_label).type(torch.float), (max_wd_idx.squeeze(-1) == wd_label).type(torch.float)) / pos_label.shape[0]
        return pos_acc, wd_acc, tot_acc

if __name__ == '__main__':

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print(device, flush=True)

    train_dataset = LSTMDataset(args.dataset_path, 'train')
    valid_dataset = LSTMDataset(args.dataset_path, 'valid')
    test_dataset = LSTMDataset(args.dataset_path, 'test')

    model = EncoderRNNWithVector(args.word_num, args.hd_size, args.pos, args.wd_ans).to(device)
    train_data = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    valid_data = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    test_data = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    val_accs = []
    test_accs = []
    for epoch in range(args.epochs):
        print(f'Epoch {epoch}')
        train_losses = []
        for i, (inputs, pos_label, wd_label) in tqdm(enumerate(train_data)):
            inputs = inputs.to(device)
            pos_label = pos_label.to(device)
            wd_label = wd_label.to(device)
            pos_output, wd_output = model(inputs)

            optimizer.zero_grad()
            loss1 = criterion(pos_output, pos_label)
            loss2 = criterion(wd_output, wd_label)
            loss = loss1 + args.alpha * loss2
            loss.backward()
            optimizer.step()

            train_loss = loss.detach().cpu().item()
            train_losses.append(train_loss)
        print(f'train_loss {torch.Tensor(train_losses).mean()}')

        pos_acc, wd_acc, tot_acc = eval_(valid_data, model)
        print('pos_acc', pos_acc)
        print('wd_acc', wd_acc)
        print('tot_acc', tot_acc)
        print('')

        val_accs.append(tot_acc)

        pos_acc, wd_acc, tot_acc = eval_(test_data, model)
        print('pos_acc', pos_acc)
        print('wd_acc', wd_acc)
        print('tot_acc', tot_acc)
        print('')

        test_accs.append(tot_acc)

    best_epoch = torch.argmax(torch.Tensor(val_accs)).item()
    print(f'test_best {test_accs[best_epoch]}')


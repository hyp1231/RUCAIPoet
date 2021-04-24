import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class RawReplace7Dataset(Dataset):
    def __init__(self, dataset_path):
        self._load_data(dataset_path)

    def _load_data(self, dataset_path):
        rpl_poets = []
        ori_poets = []
        rpl_idxs = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                rpl_poet, ori_poet, rpl_idx = line.strip().split(' ')
                rpl_idx = int(rpl_idx)
                rpl_poets.append(rpl_poet)
                ori_poets.append(ori_poet)
                rpl_idxs.append(rpl_idx)
        self.rpl_poets = rpl_poets
        self.ori_poets = ori_poets

        self.rpl_idxs = torch.LongTensor(rpl_idxs)

    def __getitem__(self, index):
        return self.rpl_poets[index], self.ori_poets[index], self.rpl_idxs[index]

    def __len__(self):
        return self.rpl_idxs.shape[0]

class NextDataset(Dataset):
    def __init__(self, dataset_path):
        self._load_data(dataset_path)
        self.word2index = np.load('word2index.npy', allow_pickle=True).item()
        
    def _load_data(self, dataset_path):
        poets = []
        flags = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                poet, flag = line.strip().split(' ')
                flag = int(flag)
                poets.append(poet)
                flags.append(flag)
        self.poets = poets
        self.flags = torch.LongTensor(flags)

    def __getitem__(self, index):
        
        indexs = [self.word2index[x] for x in self.poets[index]]
        encoding = torch.zeros((len(indexs), len(self.word2index.items())))
        for i in range(len(indexs)):
            encoding[i, indexs[i]] = 1
        
        return encoding, self.flags[index]

    def __len__(self):
        return self.flags.shape[0]
    
class LSTMDataset(RawReplace7Dataset):
    def __init__(self, dataset_path, mode):
        super(LSTMDataset, self).__init__(f'{dataset_path}.{mode}')

        self.word2index = np.load('word2index.npy', allow_pickle=True).item()
        self.rpl_poet_idxs = torch.zeros([len(self.rpl_poets), 16], dtype=torch.long)
        for i, rpl_poet in enumerate(self.rpl_poets):
            for j, wd in enumerate(rpl_poet):
                self.rpl_poet_idxs[i, j] = self.word2index[wd]
        print(mode, 'dataset', self.rpl_poet_idxs.shape, self.rpl_idxs.shape)

        ans2idx = {}
        with open('dataset/replace7/word.txt', 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                line = line.strip()
                if len(line) > 0:
                    ans2idx[line] = i
        self.ans_idxs = torch.zeros(len(self.rpl_poets), dtype=torch.long)
        for i, ori_poet in enumerate(self.ori_poets):
            self.ans_idxs[i] = ans2idx[ori_poet[self.rpl_idxs[i].item()]]

    def __getitem__(self, index):
        return self.rpl_poet_idxs[index], self.rpl_idxs[index], self.ans_idxs[index]


class NextSeq7Dataset(Dataset):

    def __init__(self, dataset_path,tokenizer):

        self.tokenizer = tokenizer
        self._load_data(dataset_path)

    def _load_data(self, dataset_path):
        poets = []
        labels = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                poet, label = line.strip().split(' ')
                label = int(label)

                poet = self.tokenizer.encode(poet, return_tensors="pt")
                #print(poet.shape)
                if poet.shape[1] != 18:
                    #print(poet)
                    continue
                poets.append(poet)
                labels.append(label)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.poets = torch.cat(poets, dim=0).to(device)

        self.labels = torch.LongTensor(labels).to(device)

    def __getitem__(self, index):
        return self.poets[index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]

if __name__ == '__main__':
    dataset_path = 'dataset/replace7/replace_poems_7'
    dataset = LSTMDataset(dataset_path, 'train')
    print(dataset[0])

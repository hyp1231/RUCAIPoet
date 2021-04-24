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

if __name__ == '__main__':
    dataset_path = 'dataset/replace7/replace_poems_7.txt'
    dataset = RawReplace7Dataset(dataset_path)
    print(dataset[0])

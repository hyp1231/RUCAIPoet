import argparse

from data import LSTMDataset
from utils import ratio_split

parser = argparse.ArgumentParser()
# parser.parse_args()
args = parser.parse_args()
args.dataset_path = 'dataset/replace7/replace_poems_7.txt'

dataset = LSTMDataset(args.dataset_path)
train_dataset, valid_dataset, test_dataset = ratio_split(dataset, [0.8, 0.1, 0.1])


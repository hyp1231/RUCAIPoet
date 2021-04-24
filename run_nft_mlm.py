from data import RawReplace7Dataset
from demo import api
from tqdm import tqdm

dataset = RawReplace7Dataset('dataset/replace7/replace_poems_7.test')
acc = 0
for rpl_poet, ori_poet, _ in tqdm(dataset):
    if api(rpl_poet) == ori_poet:
        acc += 1
print(acc / len(dataset))

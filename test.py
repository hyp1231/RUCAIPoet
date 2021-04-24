from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import argparse

args = argparse.ArgumentParser()
args.device = 0

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")

model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base")

sequence = f'浔阳江头夜送客，枫叶{tokenizer.mask_token}花秋瑟瑟'

input = tokenizer.encode(sequence, return_tensors="pt")

mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]

token_logits = model(input).logits
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))

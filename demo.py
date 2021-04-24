from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base")

ori_sequence = '半日余晖天色尽，一江逝水落云霞'
assert ori_sequence[7] == '，' and len(ori_sequence) >= 15
print(ori_sequence, '->')

best_prob = -100
best_poet = None

for possible_pos in list(range(7)) + list(range(8, 15)):
    ground_truth_token = ori_sequence[possible_pos]
    sequence = ori_sequence[:possible_pos] + \
               tokenizer.mask_token + \
               ori_sequence[possible_pos + 1:]
    input = tokenizer.encode(sequence, return_tensors="pt")
    mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
    token_logits = model(input).logits
    mask_token_logits = token_logits[0, mask_token_index, :]
    top_5_res = torch.topk(mask_token_logits, 5, dim=1)
    top_5_tokens = top_5_res.indices[0].tolist()
    prob = top_5_res.values[0]
    predict_token = tokenizer.decode([top_5_tokens[0]])
    predict_prob = prob[0].detach().cpu().numpy()
    if predict_token != ground_truth_token and predict_prob > best_prob:
        best_prob = predict_prob
        best_poet = sequence.replace(tokenizer.mask_token, predict_token)

print(best_poet, best_prob)

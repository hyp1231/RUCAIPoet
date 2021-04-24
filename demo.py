from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device, flush=True)

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base").to(device)

def api(input_poet):
    assert input_poet[7] == '，' and len(input_poet) >= 15
    # print(input_poet, '->')

    best_prob = -100
    best_poet = None

    for possible_pos in list(range(7)) + list(range(8, 15)):
        ground_truth_token = input_poet[possible_pos]
        sequence = input_poet[:possible_pos] + \
                tokenizer.mask_token + \
                input_poet[possible_pos + 1:]
        input = tokenizer.encode(sequence, return_tensors="pt")
        token_logits = model(input.to(device)).logits.cpu()
        mask_token_index = torch.where(input == tokenizer.mask_token_id)[1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        top_5_res = torch.topk(mask_token_logits, 5, dim=1)
        top_5_tokens = top_5_res.indices[0].cpu()
        prob = top_5_res.values[0]
        predict_token = tokenizer.decode([top_5_tokens[0]])
        predict_prob = prob[0].detach().cpu().numpy()
        if predict_token != ground_truth_token and predict_prob > best_prob:
            best_prob = predict_prob
            best_poet = sequence.replace(tokenizer.mask_token, predict_token)
        # print(best_poet, best_prob)
    return best_poet

if __name__ == '__main__':
    ori_sequence = '半日余晖天色尽，一江逝水落云霞'
    api(ori_sequence)

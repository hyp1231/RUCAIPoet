from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(device, flush=True)

tokenizer = AutoTokenizer.from_pretrained("ethanyt/guwenbert-base")
model = AutoModelForMaskedLM.from_pretrained("ethanyt/guwenbert-base").to(device)

def api(input_poet, max_iter):
    poets = [input_poet]
    print(input_poet, flush=True)
    for i in range(max_iter):
        best_prob = -100
        best_poet = None

        for possible_pos in range(len(input_poet)):
            if (input_poet[possible_pos] in ['，', '。']):
                continue
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
            if predict_token != ground_truth_token and predict_token != '[SEP]' and predict_prob > best_prob:
                best_prob = predict_prob
                best_poet = sequence.replace(tokenizer.mask_token, predict_token)
        # print(best_poet, best_prob)
        poets.append(best_poet)
        input_poet = best_poet
    # print(poets)
    return poets

if __name__ == '__main__':
    ori_sequence = '半日余晖天色尽，一江逝水落云霞。'
    # ori_sequence = '一堆东西天上飞，东一堆来西一堆。'
    # ori_sequence = '人言创业世事艰，我谓琼花需百炼。百年风云锤意志，千樽清酒释苦咸。世变无穷难预料，我自横刀立马前。'
    api(ori_sequence, 30)

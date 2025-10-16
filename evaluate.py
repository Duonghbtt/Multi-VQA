#Tính metric trên test set
import torch
from utils.metric import compute_bleu, compute_rouge, compute_meteor

def evaluate(model, dataloader, vocab, device):
    model.eval()
    results = []
    with torch.no_grad():
        for imgs, questions, answers in dataloader:
            imgs, questions = imgs.to(device), questions.to(device)
            preds = model.generate(imgs, questions)
            for pred, target in zip(preds, answers):
                pred_text = " ".join([vocab.idx2word[i.item()] for i in pred])
                target_text = " ".join([vocab.idx2word[i.item()] for i in target])
                bleu = compute_bleu(target_text.split(), pred_text.split())
                rouge = compute_rouge(target_text, pred_text)
                meteor = compute_meteor(target_text, pred_text)
                results.append({"bleu": bleu, "rouge": rouge, "meteor": meteor})
    # Trung bình
    avg = {k: sum([r[k] for r in results])/len(results) for k in results[0]}
    print("Average metrics:", avg)
    return avg
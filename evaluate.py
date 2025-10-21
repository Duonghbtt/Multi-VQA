#Tính metric trên test set
import torch
from utils.metric import compute_bleu, compute_rouge, compute_meteor
from inference import Vocabulary
from utils.dataset_utils import MultiVQADataset
from models.multimodal_model import VQAModel
from torch.utils.data import DataLoader

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
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary()

    all_texts = []
    test_dataset = MultiVQADataset(r"D:\VQA\data\test", "test", transform=None)
    for item in test_dataset:
        all_texts.append(item["description"])
        for conv in item["conversations"]:
            all_texts.append(conv["content"])
    vocab.build_vocab(all_texts)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 3️⃣ Load model checkpoint
    model = VQAModel.load_from_checkpoint("checkpoints/best_model.pt")
    model = model.to(device)

    # 4️⃣ Evaluate
    print("🚀 Evaluating model on test set...")
    metrics = evaluate(model, test_loader, vocab, device)

    # 5️⃣ Ghi log
    with open("results/test_metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print("✅ Evaluation complete. Metrics saved to results/test_metrics.txt")
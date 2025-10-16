# BLEU, ROUGE, METEOR
# Tính BLEU, ROUGE, METEOR cho generative output
# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score

# BLEU
def compute_bleu(reference, hypothesis):
    """
    reference, hypothesis: list of tokens
    """
    return sentence_bleu([reference], hypothesis)

# ROUGE
rouge = Rouge()
def compute_rouge(reference, hypothesis):
    """
    reference, hypothesis: strings
    """
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

# METEOR
def compute_meteor(reference, hypothesis):
    """
    reference, hypothesis: strings
    """
    return single_meteor_score(reference, hypothesis)

# ====== Test nhanh ======
if __name__ == "__main__":
    ref = "Ủy ban Nhân dân quận Ngũ Hành Sơn tổ chức kỳ thi học sinh giỏi.".split()
    hyp = "Ủy ban nhân dân quận Ngũ Hành Sơn tổ chức kỳ thi học sinh giỏi".split()

    print("BLEU:", compute_bleu(ref, hyp))
    print("ROUGE:", compute_rouge(" ".join(ref), " ".join(hyp)))
    print("METEOR:", compute_meteor(" ".join(ref), " ".join(hyp)))

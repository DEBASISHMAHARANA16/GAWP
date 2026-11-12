"""Evaluate a fill-mask model: Top-1 accuracy, Top-5 accuracy and Mean Reciprocal Rank.

Extend TEST_SET to about 40 sentences and fix it BEFORE the final run.
Each item is (sentence, acceptable answers). Answers are compared in lower case.
"""
from fill_mask import MODEL_NAME, fill_mask, load_fill_mask

TEST_SET = [
    ("The capital of France is [MASK].", ("paris",)),
    ("Tokyo is the capital of [MASK].", ("japan",)),
    ("Two plus two equals [MASK].", ("four",)),
    ("A week has seven [MASK].", ("days",)),
    ("The sky is [MASK] on a clear day.", ("blue",)),
    ("The Sun rises in the [MASK].", ("east",)),
    ("Birds can [MASK] in the sky.", ("fly",)),
    ("The doctor examined the [MASK] carefully.", ("patient",)),
    ("She poured a cup of hot [MASK].", ("tea", "coffee", "water", "chocolate", "milk")),
    ("The cat chased the [MASK] across the garden.", ("mouse", "dog", "rabbit", "bird", "squirrel")),
    ("Students attend [MASK] to learn new subjects.", ("school", "class", "classes", "college", "university")),
    ("The teacher asked the students to open their [MASK].", ("books", "notebooks", "textbooks")),
    ("India won the cricket [MASK] last year.", ("match", "series", "title", "tournament")),
    ("Parliament passed the new [MASK].", ("law", "bill", "act", "legislation", "budget")),
    ("The bank approved the [MASK] application.", ("loan", "mortgage", "credit")),
    ("The patient was given [MASK] to reduce the fever.", ("medicine", "medication", "antibiotics", "aspirin")),
]

K = 5


def evaluate(unmasker, test_set, k=K):
    top1 = topk = 0
    reciprocal_ranks = []
    for sentence, answers in test_set:
        predictions = fill_mask(unmasker, sentence, top_k=k)[0]
        words = [p["token_str"].strip().lower() for p in predictions]
        rank = next((i for i, w in enumerate(words, 1) if w in answers), None)
        top1 += rank == 1
        topk += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        print(f"{sentence:55s} predicted={words[0]:12s} rank={rank}")
    n = len(test_set)
    print(f"\nModel: {MODEL_NAME}   N = {n}")
    print(f"Top-1 accuracy : {top1 / n:.3f}")
    print(f"Top-{k} accuracy : {topk / n:.3f}")
    print(f"MRR@{k}         : {sum(reciprocal_ranks) / n:.3f}")


if __name__ == "__main__":
    evaluate(load_fill_mask(), TEST_SET)

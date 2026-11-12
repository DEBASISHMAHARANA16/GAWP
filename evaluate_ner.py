"""Evaluate NER with entity-level exact matching (text AND type must both be correct).

Compares four aggregation strategies (simple, first, average, max). For each it reports
micro precision / recall / F1, per-type scores and macro-F1.
Extend TEST_SET to about 40 sentences and fix it BEFORE the final run.
Each item is (sentence, [(entity text, type), ...]). Avoid repeating an entity in one sentence.
"""
from collections import Counter

from ner import MODEL_NAME, extract_entities, load_ner

TEST_SET = [
    ("Sundar Pichai is the chief executive of Google.", [("Sundar Pichai", "PER"), ("Google", "ORG")]),
    ("Narendra Modi visited Tokyo last week.", [("Narendra Modi", "PER"), ("Tokyo", "LOC")]),
    ("Apple opened a new store in Mumbai.", [("Apple", "ORG"), ("Mumbai", "LOC")]),
    ("The University of Oxford is located in England.", [("University of Oxford", "ORG"), ("England", "LOC")]),
    ("Elon Musk founded SpaceX in California.", [("Elon Musk", "PER"), ("SpaceX", "ORG"), ("California", "LOC")]),
    ("Microsoft announced a partnership with OpenAI in Seattle.", [("Microsoft", "ORG"), ("OpenAI", "ORG"), ("Seattle", "LOC")]),
    ("Sachin Tendulkar was born in Mumbai, India.", [("Sachin Tendulkar", "PER"), ("Mumbai", "LOC"), ("India", "LOC")]),
    ("Angela Merkel met Emmanuel Macron in Berlin.", [("Angela Merkel", "PER"), ("Emmanuel Macron", "PER"), ("Berlin", "LOC")]),
    ("Amazon has a large office in Hyderabad.", [("Amazon", "ORG"), ("Hyderabad", "LOC")]),
    ("The World Health Organization is based in Geneva.", [("World Health Organization", "ORG"), ("Geneva", "LOC")]),
    ("Bhubaneswar is the capital of Odisha.", [("Bhubaneswar", "LOC"), ("Odisha", "LOC")]),
    ("Tata Motors reported higher sales in Pune.", [("Tata Motors", "ORG"), ("Pune", "LOC")]),
    ("Lionel Messi joined Inter Miami in Florida.", [("Lionel Messi", "PER"), ("Inter Miami", "ORG"), ("Florida", "LOC")]),
    ("The Reserve Bank of India raised interest rates on Friday.", [("Reserve Bank of India", "ORG")]),
    ("Marie Curie worked at the University of Paris.", [("Marie Curie", "PER"), ("University of Paris", "ORG")]),
    ("Infosys hired engineers from Bengaluru and Chennai.", [("Infosys", "ORG"), ("Bengaluru", "LOC"), ("Chennai", "LOC")]),
    ("Barack Obama spoke at the United Nations in New York.", [("Barack Obama", "PER"), ("United Nations", "ORG"), ("New York", "LOC")]),
    ("Rahul Dravid coached the Indian cricket team.", [("Rahul Dravid", "PER"), ("Indian", "MISC")]),
    ("The French government criticised the new law.", [("French", "MISC")]),
    ("The weather was pleasant and the roads were clear.", []),
]


def score_counts(gold, pred):
    """Per-type true positives, false positives and false negatives for one sentence."""
    g, p = Counter(gold), Counter(pred)
    stats = {}
    for etype in {t for _, t in list(g) + list(p)}:
        gt = Counter({k: v for k, v in g.items() if k[1] == etype})
        pt = Counter({k: v for k, v in p.items() if k[1] == etype})
        tp = sum((gt & pt).values())
        stats[etype] = (tp, sum(pt.values()) - tp, sum(gt.values()) - tp)
    return stats


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


STRATEGIES = ["simple", "first", "average", "max"]


def evaluate(recognizer, test_set, aggregation):
    """Score one aggregation strategy; print only the sentences with errors."""
    totals = {}
    print(f"\n=== aggregation = {aggregation} ===")
    for sentence, gold in test_set:
        pred = [(e["text"], e["label"])
                for e in extract_entities(recognizer, sentence, aggregation=aggregation)]
        for etype, (tp, fp, fn) in score_counts(gold, pred).items():
            t = totals.setdefault(etype, [0, 0, 0])
            t[0] += tp; t[1] += fp; t[2] += fn
        missed = sorted(set(gold) - set(pred))
        extra = sorted(set(pred) - set(gold))
        if missed or extra:
            print(f"{sentence:62s} missed={missed} extra={extra}")

    print(f"{'Type':6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'Support':>8s}")
    f1s = []
    for etype in sorted(totals):
        tp, fp, fn = totals[etype]
        p, r, f = prf(tp, fp, fn)
        f1s.append(f)
        print(f"{etype:6s} {p:6.3f} {r:6.3f} {f:6.3f} {tp + fn:8d}")
    tp, fp, fn = (sum(v[i] for v in totals.values()) for i in range(3))
    p, r, f = prf(tp, fp, fn)
    macro = sum(f1s) / len(f1s)
    print(f"Micro P: {p:.3f}  R: {r:.3f}  F1: {f:.3f}   Macro F1: {macro:.3f}")
    return p, r, f, macro


if __name__ == "__main__":
    recognizer = load_ner()
    print(f"Model: {MODEL_NAME}   N = {len(TEST_SET)} sentences")
    summary = {s: evaluate(recognizer, TEST_SET, s) for s in STRATEGIES}
    print("\nSummary (strict entity-level)")
    print(f"{'Strategy':10s} {'Micro P':>8s} {'Micro R':>8s} {'Micro F1':>9s} {'Macro F1':>9s}")
    for name, (p, r, f, m) in summary.items():
        print(f"{name:10s} {p:8.3f} {r:8.3f} {f:9.3f} {m:9.3f}")

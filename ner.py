"""Named Entity Recognition (NER) with a pretrained token-classification model.

Usage:
    python ner.py
    python ner.py --text "Sundar Pichai is the chief executive of Google."
    python ner.py --text "Messi joined Inter Miami in Florida." --min-score 0.90
"""
import argparse

from transformers import pipeline

MODEL_NAME = "dslim/bert-base-NER"   # BERT-base (cased) fine-tuned on CoNLL-2003


def load_ner(model_name=MODEL_NAME, device=-1, aggregation="first"):
    """Load the model once; reuse the returned pipeline for every text.

    aggregation decides how word pieces are merged into entities:
      "first"   - a word takes the tag of its first piece (a word is never split)
      "average" / "max" - a word takes the tag chosen from all its pieces
      "simple"  - pieces are grouped directly; can fragment unusual names
    """
    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy=aggregation,
        device=device,
    )


def normalise_label(label):
    """Turn 'B-PER', 'I-ORG', 'MIS' etc. into one of PER / ORG / LOC / MISC."""
    label = label.upper().split("-")[-1]
    return "MISC" if label.startswith("MIS") else label


def validate_input(recognizer, text, min_score=0.0):
    """Raise ValueError for unusable input."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1.")
    limit = min(getattr(recognizer.tokenizer, "model_max_length", 512), 512) - 2
    if len(recognizer.tokenizer.tokenize(text)) > limit:
        raise ValueError(f"Text is longer than {limit} tokens; split it into passages.")


def extract_entities(recognizer, text, min_score=0.0, aggregation=None):
    """Return entities as dicts: text, label, score, start, end (sorted by position).

    aggregation (optional) overrides the strategy chosen in load_ner for this call.
    The entity text is cut from the ORIGINAL text using the character offsets,
    so it never contains word-piece markers such as '##'.
    """
    validate_input(recognizer, text, min_score)
    kwargs = {"aggregation_strategy": aggregation} if aggregation else {}
    entities = []
    for item in recognizer(text, **kwargs):
        score = float(item["score"])
        if score >= min_score:
            start, end = int(item["start"]), int(item["end"])
            entities.append({
                "text": text[start:end].strip(),
                "label": normalise_label(item["entity_group"]),
                "score": score,
                "start": start,
                "end": end,
            })
    return sorted(entities, key=lambda e: e["start"])


def annotate(text, entities):
    """Return the text with entities marked inline, e.g. [Google | ORG]."""
    marked = text
    for e in sorted(entities, key=lambda e: e["start"], reverse=True):
        marked = (marked[:e["start"]] + f"[{marked[e['start']:e['end']]} | {e['label']}]"
                  + marked[e["end"]:])
    return marked


def main():
    parser = argparse.ArgumentParser(description="Named entity recognition demo")
    parser.add_argument("--text", default="Sundar Pichai is the chief executive of Google.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--aggregation", default="first",
                        choices=["simple", "first", "average", "max"])
    parser.add_argument("--device", type=int, default=-1, help="-1 = CPU, 0 = first GPU")
    args = parser.parse_args()

    recognizer = load_ner(args.model, args.device, args.aggregation)
    entities = extract_entities(recognizer, args.text, args.min_score)

    print(f"\n{annotate(args.text, entities)}\n")
    if not entities:
        print("No entities found.")
    for e in entities:
        print(f"{e['text']:25s} {e['label']:5s} {e['score']:.4f}  [{e['start']}:{e['end']}]")


if __name__ == "__main__":
    main()

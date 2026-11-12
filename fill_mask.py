"""Fill-Mask: predict masked words in a sentence with a pretrained masked language model.

Usage:
    python fill_mask.py
    python fill_mask.py --text "The capital of France is [MASK]." --top-k 5
    python fill_mask.py --text "She drank a cup of hot [MASK]." --targets tea,coffee,milk
"""
import argparse

from transformers import pipeline

MODEL_NAME = "bert-base-uncased"


def load_fill_mask(model_name=MODEL_NAME, device=-1):
    """Load the model once; reuse the returned pipeline for every sentence."""
    return pipeline("fill-mask", model=model_name, device=device)


def validate_input(text, mask_token, targets=None):
    """Return the number of mask tokens; raise ValueError for invalid input."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    count = text.count(mask_token)
    if count == 0:
        raise ValueError(f"Text must contain the mask token {mask_token}.")
    if targets is not None:
        if any(not t.strip() for t in targets):
            raise ValueError("Target words cannot be empty.")
        if len({t.strip().casefold() for t in targets}) != len(targets):
            raise ValueError("Target words must be unique.")
    return count


def fill_mask(unmasker, text, top_k=5, targets=None):
    """Return one ranked list of predictions for every mask in the text.

    Each prediction is a dict with: score, token, token_str, sequence.
    """
    mask_token = unmasker.tokenizer.mask_token          # [MASK] or <mask>
    text = text.strip().replace("[MASK]", mask_token)   # accept [MASK] for any model
    count = validate_input(text, mask_token, targets)

    kwargs = {"top_k": top_k}
    if targets:
        kwargs["targets"] = [t.strip() for t in targets]
    outputs = unmasker(text, **kwargs)

    # One mask -> list of dicts; several masks -> list of lists. Normalise to lists.
    return [outputs] if count == 1 else outputs


def main():
    parser = argparse.ArgumentParser(description="Fill-mask demo")
    parser.add_argument("--text", default="The capital of France is [MASK].")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--targets", default=None,
                        help="Comma-separated candidate words to compare")
    parser.add_argument("--device", type=int, default=-1, help="-1 = CPU, 0 = first GPU")
    args = parser.parse_args()

    targets = args.targets.split(",") if args.targets else None
    unmasker = load_fill_mask(args.model, args.device)
    results = fill_mask(unmasker, args.text, args.top_k, targets)

    for position, predictions in enumerate(results, start=1):
        print(f"\nMask {position}")
        for rank, item in enumerate(predictions, start=1):
            print(f"{rank}. {item['token_str'].strip():12s} {item['score']:.4f}  "
                  f"{item['sequence']}")


if __name__ == "__main__":
    main()

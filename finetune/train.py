"""Fine-tune embeddinggemma-300m into the two-tower store matcher.

Step 7 of store_matching.md. Training pairs come from the db (steps 4-6 fill
embed_training_data); the score is the unnormalized dot product of the two
towers and the loss is BCE against the label probability:

    logit = E_store(store) . E_ing(ingredient)      P(stocked) = sigmoid(logit)

One tower, two prompt prefixes -- the same prefixes baseline_eval.py uses, so
the before/after numbers are comparable. Evaluation is the hand-labelled
eval_pairs.csv and nothing else: every training label is the LLM's opinion, so
scoring on held-out training pairs would only measure how well we copied it.

    uv run python scripts/store_descriptions.py   # steps 5-6, fill the db
    uv run python scripts/label_pairs.py
    uv run python finetune/train.py               # from the repo root

Prints the zero-shot scoreboard, trains, prints it again. If the second is not
clearly better than the first, store_matching.md says do not ship the fine-tune.
"""
import argparse

import duckdb
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sentence_transformers import (SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from unsloth import FastSentenceTransformer, is_bf16_supported

import baseline_eval as be
import settings

OUT = "finetune/embeddinggemma_store_lora"

# The gloss, not the raw ingredient string -- a 300m encoder cannot tell what
# "urad dal" is, and the eval side embeds the gloss too.
SQL = """
SELECT e.expansion, s.description, d.label
FROM embed_training_data d
JOIN ingredient_expansions e ON e.ingredient_id = d.ingredient_id
JOIN training_stores s ON s.id = d.training_store_id
"""

# The prompts trainer arg prepends a fixed string per column; be's prefixes are
# format strings whose placeholder is at the end, so .format("") is that string.
PROMPTS = {"ingredient": be.ING_PREFIX.format(""),
           "store": be.STORE_PREFIX.format("")}


class DotBCELoss(torch.nn.Module):
    """BCE on the raw dot product of the two towers.

    Not cosine: a well-stocked supermarket should be able to say so with a
    larger embedding norm, and normalizing throws exactly that away.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sentence_features, labels):
        a, b = (self.model(f)["sentence_embedding"] for f in sentence_features)
        return F.binary_cross_entropy_with_logits(
            (a * b).sum(-1), labels.float())


def load_training_data(db):
    rows = duckdb.connect(db, read_only=True).execute(SQL).fetchall()
    if not rows:
        raise SystemExit(f"{db}: embed_training_data is empty -- run steps 5-6 "
                         f"of finetune/store_matching.md first.")
    labels = np.array([r[2] for r in rows])
    bad = np.isin(labels, be.LABELS, invert=True)
    if bad.any():
        raise SystemExit(f"{db}: bad label {labels[bad][0]!r} in embed_training_data")
    return Dataset.from_dict({
        "ingredient": [r[0] for r in rows],
        "store": [r[1] for r in rows],
        # The name "label" is what SentenceTransformerTrainer hands the loss.
        "label": be.P[np.argmax(labels[:, None] == be.LABELS, axis=-1)],
    })


def score(model, gloss, descs):
    """The (ingredient, store) logit matrix, scored the way training does."""
    with torch.no_grad():
        ing = model.encode(gloss, prompt=PROMPTS["ingredient"])
        store = model.encode(descs, prompt=PROMPTS["store"])
    return np.asarray(ing) @ np.asarray(store).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=settings.DB)
    ap.add_argument("--pairs", default=be.PAIRS)
    ap.add_argument("--stores", default=be.STORES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    train = load_training_data(args.db)
    _, gloss, names, descs, truth = be.load_eval(args.pairs, args.stores)
    print(f"{len(train)} training pairs from {args.db}\n")

    model = FastSentenceTransformer.from_pretrained(
        model_name="unsloth/embeddinggemma-300m",
        max_seq_length=512,   # glosses are one line, store blurbs two
        full_finetuning=False,
    )

    print("=== zero-shot (this is the number to beat) ===")
    be.report(score(model, gloss, descs), truth, len(names))

    model = FastSentenceTransformer.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        task_type="FEATURE_EXTRACTION",
    )

    SentenceTransformerTrainer(
        model=model,
        train_dataset=train,
        loss=DotBCELoss(model),
        args=SentenceTransformerTrainingArguments(
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            logging_steps=10,
            prompts=PROMPTS,
            bf16=is_bf16_supported(),
            report_to="none",
            output_dir="finetune/output",
            # ponytail: no eval_strategy -- the only honest eval set is the 150
            # hand-labelled pairs, and running it every N steps would invite
            # picking the checkpoint that happens to score best on them.
        ),
    ).train()

    print("\n=== fine-tuned ===")
    be.report(score(model, gloss, descs), truth, len(names))

    model.save_pretrained(args.out)
    model.tokenizer.save_pretrained(args.out)
    print(f"\nLoRA adapters -> {args.out}")


if __name__ == "__main__":
    main()

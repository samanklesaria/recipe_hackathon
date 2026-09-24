"""Fine-tune embeddinggemma-300m into the two-tower store matcher.
"""
import csv
import duckdb
import fire
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.base.modules import Normalize
import settings
from scaled_normalize import ScaledNormalize
import store_llm

PROMPTS = {"ingredient": store_llm.ING_PREFIX, "store": store_llm.STORE_PREFIX}

OUT = "finetune/embeddinggemma_store_lora"

# Store blurbs that state outright what they do and don't carry.
EXPLICIT_CSV = "finetune/explicit_store_examples.csv"

# Hand-checked grid: ingredients down the rows, training-store names across the
# top, PROBS words in the cells.
EVAL_CSV = "finetune/eval_pairs.csv"

# The gloss, not the raw ingredient string -- a 300m encoder cannot tell what
# "urad dal" is, and the eval side embeds the gloss too.
SQL = """
SELECT e.expansion, s.description, d.label
FROM embed_training_data d
JOIN ingredient_expansions e ON e.ingredient_id = d.ingredient_id
JOIN training_stores s ON s.id = d.training_store_id
"""

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

PROBS = {"always": 0.95, "usually": 0.80, "sometimes": 0.20, "never": 0.05}

def load_training_data(db):
    rows = duckdb.connect(db, read_only=True).execute(SQL).fetchall()
    if not rows:
        raise SystemExit(f"{db}: embed_training_data is empty")
    return Dataset.from_dict({
        "ingredient": [r[0] for r in rows],
        "store": [r[1] for r in rows],
        "label": [PROBS[r[2]] for r in rows]
    })


def load_explicit_data(path):
    ds = Dataset.from_csv(path)
    return ds.map(
        lambda r: {"ingredient": r["item"], "store": r["store"],
                   # same ceiling as PROBS: hard 0/1 targets send logits to inf
                   "label": PROBS["always"] if r["has"] else PROBS["never"]},
        remove_columns=ds.column_names)


def load_eval_pairs(db, path=EVAL_CSV):
    """The wide hand-checked grid, melted into the same three columns.

    Columns are store *names*; training_stores stores the whole "Name: blurb"
    line, so match on the part before the colon.
    """
    con = duckdb.connect(db, read_only=True)
    descs = {d.split(":")[0]: d for (d,) in con.execute(
        "SELECT description FROM training_stores").fetchall()}
    glosses = dict(con.execute(
        "SELECT i.description, e.expansion FROM ingredients i"
        " JOIN ingredient_expansions e ON e.ingredient_id = i.id").fetchall())
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    unknown = {c for c in rows[0] if c != "ingredient"} - descs.keys()
    if unknown:
        raise SystemExit(f"{path}: not in training_stores: {sorted(unknown)}")
    out = {"ingredient": [], "store": [], "label": []}
    for row in rows:
        ing = row.pop("ingredient")
        for store, label in row.items():
            out["ingredient"].append(glosses.get(ing, ing))
            out["store"].append(descs[store])
            out["label"].append(PROBS[label])
    return Dataset.from_dict(out)


def score(model, gloss, descs):
    """The (ingredient, store) logit matrix, scored the way training does."""
    with torch.no_grad():
        ing = model.encode(gloss, prompt=PROMPTS["ingredient"])
        store = model.encode(descs, prompt=PROMPTS["store"])
    return np.asarray(ing) @ np.asarray(store).T


def main(db=settings.DB, out=OUT, epochs=1.0, batch_size=32, lr=2e-5):
    split = load_training_data(db).train_test_split(
        test_size=0.1, seed=3407)
    explicit = load_explicit_data(EXPLICIT_CSV).train_test_split(
        test_size=0.1, seed=3407)
    checked = load_eval_pairs(db).train_test_split(test_size=0.1, seed=3407)

    # Not unsloth: without bf16 (Turing) it keeps the base weights in fp16 and
    # turns off loss scaling, so the backward pass overflows to NaN.
    model = SentenceTransformer("unsloth/embeddinggemma-300m")
    model.max_seq_length = 512   # glosses are one line, store blurbs two
    assert isinstance(model[-1], Normalize)
    model[-1] = ScaledNormalize().to(model.device)
    model.add_adapter(LoraConfig(
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    ))

    SentenceTransformerTrainer(
        model=model,
        train_dataset={"db": split["train"], "explicit": explicit["train"],
                       "checked": checked["train"]},
        # ponytail: no evaluator -- eval_dataset runs DotBCELoss held out and
        # logs eval_loss. Add EmbeddingSimilarityEvaluator(main_similarity=
        # SimilarityFunction.DOT_PRODUCT) if you want Spearman too, but it
        # ignores `prompts`, so bake the prefixes into the strings first.
        eval_dataset={"db": split["test"], "explicit": explicit["test"],
                      "checked": checked["test"]},
        loss=DotBCELoss(model),
        args=SentenceTransformerTrainingArguments(
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            logging_steps=10,
            prompts=PROMPTS,
            learning_rate_mapping={r"beta": 1e-2},
            bf16=torch.cuda.is_bf16_supported(),
            report_to="tensorboard",
            output_dir="finetune/output",
            eval_strategy = "steps",
            eval_steps = 5,
        ),
    ).train()

    model.save_pretrained(out)
    print(f"\nLoRA adapters -> {out}")


if __name__ == "__main__":
    fire.Fire(main)

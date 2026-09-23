"""Merge the store-matcher LoRA and convert it to a GGUF for llama.cpp.

Not unsloth's save_pretrained_gguf: it converts only the inner transformer and
never passes --sentence-transformers-dense-modules, so the two Dense
projections (fine-tuned here too) silently go missing.

beta cannot go in the GGUF -- llama.cpp's graph has no op that reads it -- so
it is printed for the client to apply: serve unnormalized embeddings and scale
them by ||x|| ** (beta - 1).
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

LORA = "finetune/embeddinggemma_store_lora"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp checkout")
    ap.add_argument("--lora", default=LORA)
    ap.add_argument("--merged", default="finetune/embeddinggemma_store_merged")
    ap.add_argument("--outfile", default="finetune/embeddinggemma_store.gguf")
    ap.add_argument("--outtype", default="q8_0")
    args = ap.parse_args()

    # SentenceTransformer(lora) attaches the adapter but leaves every lora_B at
    # zero -- the trained LoRA silently vanishes (Dense and beta do load). peft
    # reads the same file correctly, so rebuild the transformer through it.
    m = SentenceTransformer(args.lora, device="cpu", trust_remote_code=True)
    base = json.loads((Path(args.lora) / "adapter_config.json").read_text())[
        "base_model_name_or_path"]
    m[0].model = PeftModel.from_pretrained(
        AutoModel.from_pretrained(base), args.lora).merge_and_unload()
    m.save_pretrained(args.merged)
    # The Gemma converter needs the sentencepiece model, which save_pretrained
    # does not write; without it it falls back to a BPE vocab and asserts.
    shutil.copy(hf_hub_download(base, "tokenizer.model"), args.merged)

    llama_cpp = Path(args.llama_cpp)
    subprocess.run([sys.executable, llama_cpp / "convert_hf_to_gguf.py",
                    args.merged, "--outfile", args.outfile,
                    "--outtype", args.outtype,
                    "--sentence-transformers-dense-modules"], check=True)

    # The failure this script exists to avoid is silent, so check for it.
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf
    names = [t.name for t in gguf.GGUFReader(args.outfile).tensors]
    if not any("dense" in n for n in names):
        raise SystemExit(f"{args.outfile}: no dense tensors, projections lost")

    print(f"\n{args.outfile}\nexport RECIPE_EMBED_BETA={m[-1].beta.item()}")


if __name__ == "__main__":
    main()

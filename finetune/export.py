"""Merge the store-matcher LoRA and convert it to a GGUF for llama.cpp.

Not unsloth's save_pretrained_gguf: it converts only the inner transformer and
never passes --sentence-transformers-dense-modules, so the two Dense
projections (fine-tuned here too) silently go missing.

beta cannot go in the GGUF -- llama.cpp's graph has no op that reads it -- so
it is printed for the client to apply: serve unnormalized embeddings and scale
them by ||x|| ** (beta - 1).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import fire
from huggingface_hub import hf_hub_download
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

LORA = "finetune/embedding_store_lora"


def main(llama_cpp, lora=LORA, merged="finetune/embeddinggemma_store_merged",
         outfile="finetune/embeddinggemma_store.gguf", outtype="q8_0"):
    """llama_cpp is the path to a llama.cpp checkout."""
    # SentenceTransformer(lora) attaches the adapter but leaves every lora_B at
    # zero -- the trained LoRA silently vanishes (Dense and beta do load). peft
    # reads the same file correctly, so rebuild the transformer through it.
    m = SentenceTransformer(lora, device="cpu", trust_remote_code=True)
    base = json.loads((Path(lora) / "adapter_config.json").read_text())[
        "base_model_name_or_path"]
    m[0].model = PeftModel.from_pretrained(
        AutoModel.from_pretrained(base), lora).merge_and_unload()
    m.save_pretrained(merged)
    # The Gemma converter needs the sentencepiece model, which save_pretrained
    # does not write; without it it falls back to a BPE vocab and asserts.
    shutil.copy(hf_hub_download(base, "tokenizer.model"), merged)

    checkout = Path(llama_cpp)
    subprocess.run([sys.executable, checkout / "convert_hf_to_gguf.py",
                    merged, "--outfile", outfile,
                    "--outtype", outtype,
                    "--sentence-transformers-dense-modules"], check=True)

    # The failure this script exists to avoid is silent, so check for it.
    sys.path.insert(0, str(checkout / "gguf-py"))
    import gguf
    names = [t.name for t in gguf.GGUFReader(outfile).tensors]
    if not any("dense" in n for n in names):
        raise SystemExit(f"{outfile}: no dense tensors, projections lost")

    print(f"\n{outfile}\nexport RECIPE_EMBED_BETA={m[-1].beta.item()}")


if __name__ == "__main__":
    fire.Fire(main)

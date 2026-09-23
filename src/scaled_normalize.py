import torch
from sentence_transformers.base.modules.module import Module


class ScaledNormalize(Module):
    """(x / ||x||) * ||x||**beta, beta trainable from 0 -- plain Normalize at init.

    beta = 1 passes the raw norm through, so training decides how much of the
    input's norm survives into the dot product.
    """

    def __init__(self):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(()))

    def forward(self, features):
        x = features["sentence_embedding"]
        n = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        features["sentence_embedding"] = x / n * n ** self.beta
        return features

    def save(self, output_path, *args, safe_serialization=True, **kwargs):
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(cls, model_name_or_path, subfolder="", token=None, cache_folder=None,
             revision=None, local_files_only=False, **kwargs):
        return cls.load_torch_weights(
            model_name_or_path=model_name_or_path, subfolder=subfolder,
            token=token, cache_folder=cache_folder, revision=revision,
            local_files_only=local_files_only, model=cls())

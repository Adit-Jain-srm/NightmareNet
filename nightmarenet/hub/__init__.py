from transformers import PyTorchModelHubMixin

class NightmareNetHubMixin(PyTorchModelHubMixin):
    def push_to_hub(self, repo_id, **kwargs):
        super().push_to_hub(repo_id, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

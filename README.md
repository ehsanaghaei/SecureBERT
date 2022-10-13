# SecureBERT

[SecureBERT](https://arxiv.org/pdf/2204.02685) is a domain-specific language model to represent cybersecurity textual data.

---

# How to use SecureBERT
SecureBERT has been uploaded to [Huggingface](https://huggingface.co/ehsanaghaei/SecureBERT) framework:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")

model = AutoModelForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")
```

Or just clone the repo:

```bash
git lfs install
git clone https://huggingface.co/ehsanaghaei/SecureBERT
# if you want to clone without large files – just their pointers
# prepend your git clone with the following env var:
GIT_LFS_SKIP_SMUDGE=1
```

---
* This repo will be updated on a regular basis.

***
# References
https://arxiv.org/pdf/2204.02685

# SecureBERT

[SecureBERT](https://arxiv.org/pdf/2204.02685) is a domain-specific language model to represent cybersecurity textual data.

---

# How to use SecureBERT
SecureBERT has been uploaded to [Huggingface](https://huggingface.co/ehsanaghaei/SecureBERT) framework, You may 

```python
from transformers import RobertaTokenizer, RobertaModel
import torch

tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = RobertaModel.from_pretrained("ehsanaghaei/SecureBERT")

inputs = tokenizer("Cybersecurity is mandatory!", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

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
SecureBERT has been train on MLM task. Use the code below to predict the masked word in the given sentences:

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
model = RobertaForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")

inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# retrieve index of <mask>
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
tokenizer.decode(predicted_token_id)
```
---
* This repo will be updated on a regular basis.

***
# References
https://arxiv.org/pdf/2204.02685

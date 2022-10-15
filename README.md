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

inputs = tokenizer("This is SecureBERT!", return_tensors="pt")
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
#!pip install transformers
#!pip install torch
#!pip install tokenizers

import torch
def predict_mask(sent, tokenizer, model, topk =10, print_results = True):
    token_ids = tokenizer.encode(sent, return_tensors='pt')
    masked_position = (token_ids.squeeze() == tokenizer.mask_token_id).nonzero()
    masked_pos = [mask.item() for mask in masked_position]
    words = []
    with torch.no_grad():
        output = model(token_ids)

    last_hidden_state = output[0].squeeze()

    list_of_list = []
    for index, mask_index in enumerate(masked_pos):
        mask_hidden_state = last_hidden_state[mask_index]
        idx = torch.topk(mask_hidden_state, k=topk, dim=0)[1]
        words = [tokenizer.decode(i.item()).strip() for i in idx]
        words = [w.replace(' ','') for w in words]
        list_of_list.append(words)
        if print_results:
            print("Mask ", "Predictions : ", words)

    best_guess = ""
    for j in list_of_list:
        best_guess = best_guess + "," + j[0]

    return words


import transformers
from transformers import RobertaTokenizer, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("ehsanaghaei/SecureBERT")
model = transformers.RobertaForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")


import torch
while True:
    sent = input("Text here: \t")
    print("SecureBERT: ")
    predict_mask(sent, tokenizer, model)
     
    print("===========================\n")
```

##Samples results:
```text
Text here: 	Gathering this information may reveal opportunities for other forms of <mask>, establishing operational resources, or initial access.
SecureBERT: 
Mask  Predictions :  ['reconnaissance', 'exploitation', 'attack', 'compromise', 'phishing', 'enumeration', 'attacks', 'penetration', 'espionage', 'intrusion']
===========================

Text here: 	Information about identities may include a variety of details, including personal data as well as <mask> details such as credentials.
SecureBERT: 
Mask  Predictions :  ['authentication', 'sensitive', 'system', 'credential', 'other', 'additional', 'security', 'technical', 'account', 'secret']
===========================

Text here: 	Adversaries may also compromise sites then include <mask> content designed to collect website authentication cookies from visitors.
SecureBERT: 
Mask  Predictions :  ['malicious', 'JavaScript', 'phishing', 'iframe', 'dynamic', 'additional', 'downloadable', 'hostile', 'embedded', 'website']
===========================

Text here: 	Adversaries may also compromise sites then include malicious content designed to collect website authentication <mask> from visitors.
SecureBERT: 
Mask  Predictions :  ['credentials', 'information', 'data', 'tokens', 'details', 'cookies', 'parameters', 'codes', 'secrets', 'keys']
===========================

Text here: 	The given website may closely resemble a <mask> site in appearance and have a URL containing elements from the real site.
SecureBERT: 
Mask  Predictions :  ['real', 'legitimate', 'trusted', 'fake', 'genuine', 'live', 'phishing', 'similar', 'known', 'different']
===========================

Text here: 	Although sensitive details may be redacted, this information may contain trends regarding <mask> such as target industries, attribution claims, and successful countermeasures.
SecureBERT: 
Mask  Predictions :  ['events', 'indicators', 'activities', 'topics', 'activity', 'trends', 'factors', 'incidents', 'information', 'campaigns']
===========================

Text here: 	Adversaries may search digital <mask> data to gather actionable information.
SecureBERT: 
Mask  Predictions :  ['forensic', 'media', 'forensics', 'asset', 'image', 'security', 'form', 'identity', 'store', 'document']
===========================

Text here: 	Once credentials are obtained, they can be used to perform <mask> movement and access restricted information.
SecureBERT: 
Mask  Predictions :  ['lateral', 'laterally', 'Lateral', 'data', 'credential', 'horizontal', 'illegal', 'money', 'physical', 'unauthorized']
===========================

Text here: 	One example of this is MS14-068, which targets <mask> and can be used to forge Kerberos tickets using domain user permissions.
SecureBERT: 
Mask  Predictions :  ['Kerberos', 'authentication', 'users', 'Windows', 'administrators', 'LDAP', 'PAM', 'Samba', 'NTLM', 'AD']
```
---
* This repo will be updated on a regular basis.

***
# References
https://arxiv.org/pdf/2204.02685

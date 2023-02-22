# SecureBERT

[SecureBERT](https://arxiv.org/pdf/2204.02685) is a domain-specific language model to represent cybersecurity textual data which is trained on a large amount of in-domain text crawled from online resources. 
![image](https://user-images.githubusercontent.com/46252665/195998237-9bbed621-8002-4287-ac0d-19c4f603d919.png)

***See the presentation on [YouTube](https://www.youtube.com/watch?v=G8WzvThGG8c&t=8s)***


## SecureBERT can be used as the base model for any downstream task including text classification, NER, Seq-to-Seq, QA, etc.
* SecureBERT has demonstrated significantly higher performance in predicting masked words within the text when compared to existing models like RoBERTa (base and large), SciBERT, and SecBERT.
* SecureBERT has also demonstrated promising performance in preserving general English language understanding (representation).

---

# How to use SecureBERT
SecureBERT has been uploaded to [Huggingface](https://huggingface.co/ehsanaghaei/SecureBERT) framework. You may use the code below

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
## Fill Mask
SecureBERT has been trained on MLM. Use the code below to predict the masked word within the given sentences:

```python
#!pip install transformers
#!pip install torch
#!pip install tokenizers

import torch
import transformers
from transformers import RobertaTokenizer, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("ehsanaghaei/SecureBERT")
model = transformers.RobertaForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")

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


while True:
    sent = input("Text here: \t")
    print("SecureBERT: ")
    predict_mask(sent, tokenizer, model)
     
    print("===========================\n")
```

### Sample results:
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

Text here: 	One example of this is MS14-068, which targets <mask> and can be used to forge its tickets.
SecureBERT: 
Mask  Predictions :  ['Kerberos', 'tickets', 'RSA', 'KDC', 'LDAP']
```
---

![mlm_example](https://user-images.githubusercontent.com/46252665/195998153-f5682f7c-60a8-486d-b2c1-9ef5732c24ba.png)

SecureBERT outperforms the existing models in MLM testing conducted on a manually crafted dataset from the human readable descriptions of MITRE ATT&CK techniques and tactics.

![image](https://user-images.githubusercontent.com/46252665/195998409-1e175f94-c35e-4682-9bf4-f6bc5c5d1627.png)

![image](https://user-images.githubusercontent.com/46252665/195998407-88a8b61e-a3dd-4196-be1e-9af6c0f647f6.png)


* This repo will be updated on a regular basis.


# SecureBERT is DIFFERENT than [SecBERT](https://huggingface.co/jackaduma/SecRoBERTa?text=Email+protocol+is+called+%3Cmask%3E.). 
```
Sample Text: 	Adversaries may also compromise sites then include <mask> content designed to collect website authentication cookies from visitors.
>> SecureBERT: 
malicious |	JavaScript |	phishing |	iframe |	dynamic |	additional |	downloadable |	hostile |	embedded |	website


>> SecBERT: 
web |	exploit |	simple |	a |	social |	clipboard |	shared |	native |	malicious |	business

-------------------------------------------

Sample Text: 	One example of this is MS14-068, which targets <mask> and can be used to forge Kerberos tickets using domain user permissions.
>> SecureBERT: 
Kerberos |	authentication |	users |	Windows |	administrators |	LDAP |	PAM |	Samba |	NTLM |	AD


>> SecBERT: 
businesses |	passwords |	differ |	system |	code |	software |	sensitive |	known |	MySQL |	misconfigurations

-------------------------------------------

Sample Text: 	Paris is the <mask> of France.
>> SecureBERT: 
capital |	Republic |	Government |	province |	name |	city |	government |	language |	Capital |	Bank


>> SecBERT: 
case |	course |	leader |	peak |	Ell |	Embassy |	trademark |	Republic |	Dictionary |	ministry

-------------------------------------------

Sample Text: 	Virus causes <mask>.
>> SecureBERT: 
DoS |	crash |	reboot |	reboots |	panic |	crashes |	corruption |	DOS |	vulnerability |	XSS


>> SecBERT: 
disabled |	invisible |	vulnerabilities |	advanced |	before |	left |	too |	vector |	obviously |	vulnerable

-------------------------------------------

Sample Text: 	Sending huge amount of packets through network leads to <mask>.
>> SecureBERT: 
DoS |	congestion |	crashes |	crash |	problems |	DOS |	vulnerability |	failure |	vulnerabilities |	errors


>> SecBERT: 
monitoring |	pipes |	situations |	default |	minutes |	login |	workstations |	services |	printers |	etc

-------------------------------------------

Sample Text: 	A <mask> injection occurs when an attacker inserts malicious code into a server
>> SecureBERT: 
code |	SQL |	command |	malicious |	script |	web |	vulnerability |	server |	ql |	SQL


>> SecBERT: 
frame |	keystroke |	packet |	massive |	mass |	URL |	vector |	COM |	window |	Distributed
```
***
# References
@inproceedings{aghaei2023securebert,
  title={SecureBERT: A Domain-Specific Language Model for Cybersecurity},
  author={Aghaei, Ehsan and Niu, Xi and Shadid, Waseem and Al-Shaer, Ehab},
  booktitle={Security and Privacy in Communication Networks: 18th EAI International Conference, SecureComm 2022, Virtual Event, October 2022, Proceedings},
  pages={39--56},
  year={2023},
  organization={Springer}
}
[https://link.springer.com/chapter/10.1007/978-3-031-25538-0_3]

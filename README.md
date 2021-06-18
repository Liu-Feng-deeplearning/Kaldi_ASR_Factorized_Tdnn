# Kaldi_ASR_Factorized_Tdnn

PyTorch implementation of the Kaldi asr acoustic model of factorized tdnn. 
And this model can be used in asr or other task(eg ppg extractor for vc)

Factorized tdnn is known in nnet3 of Kaldi. And in my opnion,  
it is almost best model in Asr because of the trade between accuracy and effecient.
You can also read [paper](http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) for more details about that.

And the whole model architecture is the same as 
kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7p and kaldi/egs/swbd/s5c/local/chain/tuning/run_tdnn_7q, 
and some parameters about Kaldi's layer are writen in 
kaldi/egs/swbd/s5c/steps/libs/nnet3/xconfig/composite_layers.py

experiements about tdnnf can be seen in kaldi/egs/.../.../run_tdnn_xx

Lots of codes in this project inherit from [cvqluu/Factorized-TDNN](https://github.com/cvqluu/Factorized-TDNN/blob/master/README.md). I has to say Thanks to him.

---
## Key-Feature about TdnnF

- using 3-stage splicing instead of baseline to get wider and more effective context.  
- also factorizing the final layer to reduce parameters.
- skip connections for deep layers. As metioned in the paper, we also choose "small to large" option.
- shared dim scale dropout, the dropout masks are shared across
time, and the random scale is chosen from Uniform([1 - 2\alpha, 1 + 2\alpha])

---

## Usage

you can see test script in tdnnf_model.py and test for layer and whole model.
This is example for training of the model:

```python
import torch, yaml
from tdnnf_model import TdnnfModel

with open("hparams.yaml", encoding="utf-8") as yaml_file:
  hp = yaml.safe_load(yaml_file)  # using yaml file for hparames
model = TdnnfModel(hp)

for x, y in batch:
  pred_y = model(x)
  loss = loss_fun(pred_y, y)
  ...
  loss.backward()
  ...
  model.step_ftdnn_layers()  # Note: don`t forget it
  ...

```

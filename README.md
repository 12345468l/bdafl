# Federated-Backdoor-Attack

## Prerequisites

- Python (3.8+, is a must)
- Pytorch (1.11)
- CUDA (1.10+)
- some other packages (just conda install or pip install)

## Step to run
1. get into the directory `Focused-Flip-Federated-Backdoor-Attack/`

2. get Tiny-ImageNet dataset

   ```bash
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
   unzip tiny-imagenet-200.zip
   ```

3. run Bases.py
   ```bash
   python Bases.py 
   --defense {fedavg,ensemble-distillation,mediod-distillation,fine-tuning,mitigation-pruning,robustlr,certified-robustness,bulyan,deep-sight} 
   --config {cifar,imagenet} 
   --backdoor {ff,dba,naive,neurotoxin}
   --model {simple,resnet18}
   ```

   Hyperparameters about attack and defense baselines are mostly in `Params.py`, hyperparameters about dataset are mostly in `configs/`

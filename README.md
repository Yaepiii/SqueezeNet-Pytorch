# SqueezeNet-Pytorch
This repo is implementation for SqueezeNet(https://arxiv.org/abs/1602.07360) in pytorch. The model is in `SqueezeNet/SqueezeNet.py`.

It is tested with pytorch-1.0.

# Download data and running

```
git clone https://github.com/Yaepiii/SqueezeNet-Pytorch
cd SqueezeNet-Pytorch
pip install torch
```

or, you are in anaconda:

```
conda create -n SqueezeNet-Pytorch python=3.8
conda activate SqueezeNet-Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Training 
```
python train_SqueezeNet.py --batch_size=128 --epoch=20 --lr=0.1 --m=0.9 --version=1 --wd=0.001 
```

version1 or version2 is following:

<div align=center>
<img src="https://github.com/Yaepiii/SqueezeNet-Pytorch/assets/75295024/b9fec104-4035-4689-bd3e-57eec4697342" width="480" height="450">
</div>

This will create a 'parameter' folder to store the trained parameters.

And some trained results have been saved in 'result' floder.

Test
```
python test_SqueezeNet.py --batch_size=128 --epoch=20 --version=1
```
Note the corresponding file!

# Performance

Accuracy:

|  | version1 | version2 |
| :---: | :---: | :---: | 
| epoch=5 | 66.8 | 64.9 |
| epoch=20 | 82.2 | 82.8 |

 The squeezenet_version2_epoch20 loss plot is following:

<div align=center>
<img src="https://github.com/Yaepiii/SqueezeNet-Pytorch/assets/75295024/d1363375-0255-49a6-8f20-07a0a0c7a935" width="480" height="450">
</div>





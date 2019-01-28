## kaggle cifar10 with Pytorch

### Introduction

In this project, we use the `DPNet` with Pytorchto do the cifar10 classification. And you could also do it with other networks, such as：`Resnet18` ,  `Densent`, `ResneXt` , and so on...

### Git Project

open the terminal and run the command

```
git clone https://github.com/L1aoXingyu/kaggle-cifar10.git
```

or you can download it via the website

### Data Download
Download the data via the website (https://www.kaggle.com/c/cifar-10)

<div align=center>
<img src='https://ws1.sinaimg.cn/large/006tNbRwly1fwai15kmgvj31he13aq5k.jpg' width='500'>
</div>

makedirs `data`，and copy these data into the `data`，then running the below command to get the data what we want.

```
sudo apt-get install p7zip
cd data;
p7zip -d train.7z;
p7zip -d test.7z;
cd ..
cd utils; python preprocess.py;
```

It may cost several minutes ...

### Training
Running the code:

```
python train.py --bs=128  # cpu Training
python train.py --bs=128 --use_gpu # gpu Training
```

`bs` represent batch size, `use_gpu` means using gpu，and for other parameters, please refer to `train.py`. During the training process, the weights will be saved in `checkpoints` automatically.
### submission
Afther the training, we could load the best weights we got before, and run the below command to archive the submission.csv file. Then we just need to submit this file to the Kaggle, and we will get the score in seconds.

```
python submit.py --model_path='checkpoints/model_best.pt' --use_gpu
```

<div align=center>
<img src='https://i.loli.net/2019/01/28/5c4ed4407df48.png' width='800'>
</div>

### other
Actually, we could try to different network architecture to check the accuracy.

The table as below is just a reference。

Model	| Acc.
---|---
VGG16		| 92.64%
ResNet18		| 93.02%
ResNet50		| 93.62%
ResNet101		| 93.75%
MobileNetV2		| 94.43%
ResNeXt29(32x4d)		| 94.73%
ResNeXt29(2x64d)		| 94.82%
DenseNet121		| 95.04%
PreActResNet18		| 95.11%

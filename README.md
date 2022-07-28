# BNN VAD pytorch

本项目基于[https://github.com/itayhubara/BinaryNet.pytorch](BNN-pytorch)进行修改，目的为2分类语音数据集进行分类，若任何问题可以提issue或直接邮件联系，若引用请考虑引用原作者论文

本工程在原作基础上加入自定义数据集读取部分，且加入cuda并行部分，注释完备,修改方便~

直接运行指令参考：
```python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10```


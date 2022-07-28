import json
from pandas import value_counts
import torch
import numpy as np
import os
import pickle
import random
# import matplotlib.pyplot as plt
# TODO 检查 label 是否只有 1………………
# root = "/home/jclou/kwsprj/data/vaddata"
# val_rate = 0.02
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root),"dataset root: {} does not exist".format(root)

    VAD_class = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root,cls))]
    VAD_class.sort()
    print(VAD_class)
    class_indice = dict((k,v) for v,k in enumerate(VAD_class))
    print(class_indice)
    str_json = json.dumps(dict((val,key) for key,val in class_indice.items()), indent=4)
    # print(str_json)

    train_voice_path = []
    train_voice_label = []
    val_voice_path = []
    val_voice_label = []
    every_cls_num = [] # 每个样本的数量

    supported_file = ['.csv']

    for cls in VAD_class:
        cls_path = os.path.join(root,cls)
        voice = [os.path.join(root,cls,i) for i in os.listdir(cls_path) 
                    if os.path.splitext(i)[-1] in supported_file] # csv 组成的 文件名 list
        # for i in os.listdir(cls_path):
        #     print(os.path.splitext(i))
        #     print(os.path.splitext(i)[-1])
        voice_label = class_indice[cls] # voice 对应的 label
        # print(voice_label)
        # 结果为 0 和 1 , 格式为 int

        every_cls_num.append(len(voice))
        # print(every_cls_num)
        val_path = random.sample(voice, k=int(len(voice) * val_rate))

        # val_voice_label = random.sample(voice_label, k=int(len(voice) * val_rate) )
        for voice_path in voice:
            if voice_path in val_path:
                val_voice_path.append(voice_path)
                val_voice_label.append(voice_label)
                # val_voice_label.append(list.index())
            else:
                train_voice_path.append(voice_path)
                train_voice_label.append(voice_label)
        # print(val_voice_label)
        # print(train_voice_label)
    print("{} voice were found in the dataset.".format(sum(every_cls_num)))
    print("{} voice for training.".format(len(train_voice_path)))
    print("{} voice were found in the val dataset.".format((len(val_voice_path))))
# print(val_voice_path)
# print(val_voice_label)
# print(type(voice))
# print(len(voice))
# print("{} voice were found in the dataset.".format(sum(cls_num)))

    return train_voice_path,train_voice_label,val_voice_path,val_voice_label
# print(voice_label)
# print("{} voice were found in the dataset.".format(sum(cls_num)))
# print("{} voice for training.".format(len(train_voice_path)))
# print("{} voice were found in the dataset.".format((len(val_voice_path))))
    
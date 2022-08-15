from curses import keyname
from turtle import shape
import torch
import os
from torchsummary import summary

# file_path = '/home/jclou/kwsprj/BinaryNet.pytorch/results/model_best.pth.tar'
file_path = '/home/jclou/kwsprj/BNN-VAD/results/kws_2/model_best.pth.tar'
kws3 = '/home/jclou/kwsprj/BNN-VAD/results/kws_3/model_best.pth.tar'
os.environ['CUDA_VISIBLE_DEVICE'] = '3'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(file_path, map_location= device)
torch.save(model,"/home/jclou/kwsprj/BNN-VAD/results/kws_3/model.pth")
# model = model.to(device)
print(model.keys())
# print(model['model'])
# print(model['config'])
state_dict = model['state_dict']
# print(state_dict['features.0.weight'])
# print(state_dict['classifier.0.weight'])
# print(state_dict['features.0.bias'])
print(state_dict.keys())

# 输出：odict_keys(['features.0.weight', 'features.0.bias', 'features.2.weight', 'features.2.bias', 'features.2.running_mean', 'features.2.running_var', 'features.2.num_batches_tracked', 'features.4.weight', 'features.4.bias', 'features.6.weight', 'features.6.bias', 'features.6.running_mean', 'features.6.running_var', 'features.6.num_batches_tracked', 'features.8.weight', 'features.8.bias', 'features.9.weight', 'features.9.bias', 'features.9.running_mean', 'features.9.running_var', 'features.9.num_batches_tracked', 'features.11.weight', 'features.11.bias', 'features.12.weight', 'features.12.bias', 'features.12.running_mean', 'features.12.running_var', 'features.12.num_batches_tracked', 'features.14.weight', 'features.14.bias', 'features.16.weight', 'features.16.bias', 'features.16.running_mean', 'features.16.running_var', 'features.16.num_batches_tracked', 'classifier.0.weight', 'classifier.0.bias', 'classifier.1.weight', 'classifier.1.bias', 'classifier.1.running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked', 'classifier.3.weight', 'classifier.3.bias', 'classifier.4.weight', 'classifier.4.bias', 'classifier.4.running_mean', 'classifier.4.running_var', 'classifier.4.num_batches_tracked', 'classifier.6.weight', 'classifier.6.bias', 'classifier.7.weight', 'classifier.7.bias', 'classifier.7.running_mean', 'classifier.7.running_var', 'classifier.7.num_batches_tracked'])
# for key,value in model['model']:
#     print(key,value,sep="")
# summary(model,(1,12,32,32))
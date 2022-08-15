import torch
import torch.nn as nn
import onnx
import numpy as np
from models.kws_bnn_binary2 import kws_bnn2
from models.kws_bnn_binary3 import kws_bnn3

import os

file_path = '/home/jclou/kwsprj/BNN-VAD/results/kws_3/model_best.pth.tar'
pth = '/home/jclou/kwsprj/BNN-VAD/results/kws_3/model.pth'
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = kws_bnn3()
# model = torch.load(file_path, map_location= torch.device('cuda')) # cuda
state_dict = torch.load(file_path, map_location= torch.device('cpu')) # cpu

model.load_state_dict(state_dict['state_dict'])
# model.load_state_dict(file_path)

# model.load_state_dict(torch.load(pth))

# model.eval()

# model = torch.load(file_path, map_location= device)
# model.eval()


# input_names = ["input","conv_input","hardtanh_input","fc_input","bn_input","logSoftmax_input"]
# output_names = ["output_0","output_1"]
input = torch.randn(size = (1,1,20,6) ) 

torch.onnx.export(model,input,'/home/jclou/kwsprj/BNN-VAD/results/kws_3/model2.onnx',export_params=False,verbose=True)
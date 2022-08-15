from importlib.resources import path
import torch
import torch.nn as nn
import onnx
import numpy as np
from models.kws_bnn_binary2 import kws_bnn2
from models.kws_bnn_binary3 import kws_bnn3
from models.alexnet_binary  import AlexNetOWT_BN
from onnxsim import simplify
import os

save_path = './results/kws_3/'
file_path = './results/kws_3/model_best.pth.tar'
# pth = '/home/jclou/kwsprj/BNN-VAD/results/kws_3/model.pth'
# alexnet = '/home/jclou/kwsprj/BNN-VAD/results/alexnet_binary_100/model_best.pth.tar'

# GPU 支持
# os.environ['CUDA_VISIBLE_DEVICE'] = '3'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
model = kws_bnn3()
# model = AlexNetOWT_BN()
# model = torch.load(file_path, map_location= torch.device('cuda')) # cuda
state_dict = torch.load(file_path, map_location= torch.device('cpu')) # cpu
# state_dict = torch.load(alexnet, map_location= torch.device('cpu')) # cpu
# model.load_state_dict(torch.load(alexnet,map_location=torch.device('cpu')))
model.load_state_dict(state_dict['state_dict'])
# model.load_state_dict(file_path)
# model.load_state_dict(torch.load(pth))

model.eval()

# model = torch.load(file_path, map_location= device)
# model.eval()


input_names = ["input"]
output_names = ["output"]
input = torch.randn(size = (1,1,20,6) ) 
# alex = torch.randn(size=(1,3,224,224))

torch.onnx.export(model,input,os.path.join(save_path,'model.onnx'),export_params=False,verbose=True,input_names=input_names,output_names=output_names)

# onnx 简化
model = onnx.load(os.path.join(save_path,'model.onnx'))
model_simp, check = simplify(model)
onnx.save(onnx.shape_inference.infer_shapes(model_simp), os.path.join(save_path,'model_simple2.onnx'))
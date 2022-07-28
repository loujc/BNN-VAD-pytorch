import numpy as np
import sys
sys.path.append('../')

from input.processor.folder_processor import FolderProcessor as FolderProcessor
from input.processor.data_reader import DataReader as DataReader
from input.processor.processor_builder import ProcessorBuilder as ProcessorBuilder

builder = ProcessorBuilder()

preprocessor = builder.build()

folderProcessor = FolderProcessor(r"/home/zhangyadong/work/vad/vaddata")
label_list, label_str_list,datafile_list  = folderProcessor.generate_target_list()
label_list, feature_list = preprocessor.generate_features(label_list, label_str_list,datafile_list,fixed_length = 1792)


print(label_list)
print(feature_list)

print(type(label_list))
print(type(feature_list))
print(label_list.shape())
print(feature_list.shape())
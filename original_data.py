# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
from chainer.datasets import tuple_dataset
from chainercv import transforms
from tqdm import tqdm

def get_data(path):
    
    files = os.listdir(path)
    
    datasets = []
    labels = []
    file_names = []

    for folder, subfolders, files in os.walk(path):
        for file in tqdm(files):

            full_path = os.path.join(folder, file)

            img = Image.open(full_path)

            dataset = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.

            # dataset = transforms.resize(dataset,(500, 500))
        
            file_name, file_ext = os.path.splitext(file)

            label = np.int32(str(file_name)[-1])

            datasets.append(dataset)
            labels.append(label)
            file_names.append(file)

    return tuple_dataset.TupleDataset(datasets, labels), file_names

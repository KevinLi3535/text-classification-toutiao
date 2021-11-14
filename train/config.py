

import os
import numpy as np
import torch
import random

PARENT_DIR = 'drive/My Drive/github/text-classification-toutiao'
MODEL_DIR = os.path.join(PARENT_DIR,'model')
DATA_DIR = os.path.join(PARENT_DIR,'data')
MCFG_DIR = os.path.join(PARENT_DIR,'model_config')

config = {
    'train_file_path': os.path.join(DATA_DIR,'train.csv'),
    'test_file_path': os.path.join(DATA_DIR,'test.csv'),
    'project_dir':PARENT_DIR,
    'data_dir':DATA_DIR,
    'model_config_dir':MCFG_DIR,
    'model_dir':MODEL_DIR,
    'train_val_ratio': 0.1,
    'seed': 2021 
}

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

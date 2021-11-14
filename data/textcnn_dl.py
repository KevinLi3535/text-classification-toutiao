import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.config import *

import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from collections import Counter,defaultdict
import bz2
import jieba
from tqdm import  tqdm

def get_vocab(config):
    token_counter = Counter()
    with open(config['train_file_path'], 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc='Counting tokens', total=len(lines)):
            sent = line.split(',')[-1].strip()
            sent_cut = list(jieba.cut(sent))
            token_counter.update(sent_cut)
            # token_counter {'我': 2,'是': 5}
    
    vocab = set(token for token, _ in token_counter.most_common(config['vocab_size']))
    return vocab

def get_embedding(vocab):
    token2embedding ={}

    with bz2.open(os.path.join(config['model_config_dir'],'textcnn_config','sgns.weibo.word.bz2')) as f:
        token_vector = f.readlines()

        meta_info = token_vector[0].split()
        print(f'{meta_info[0]} tokens in embedding file in total, vector size is {meta_info[1]}')

        # '我' 0.88383 0.22222 *300
        for line in tqdm(token_vector[1:]):
            line = line.split()
            token = line[0].decode('utf8')

            vector = line[1:]

            if token in vocab:
                token2embedding[token] = [float(num) for num in vector]

        token2id = {token:idx for idx, token in enumerate(token2embedding.keys(), 4)}
        id2embedding = {token2id[token]: embedding for token, embedding in token2embedding.items()}

        PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'

        token2id[PAD] = 0
        token2id[UNK] = 1
        token2id[BOS] = 2
        token2id[EOS] = 3

        id2embedding[0] = [.0] * int(meta_info[1])
        id2embedding[1] = [.0] * int(meta_info[1])

        id2embedding[2] = np.random.random(int(meta_info[1])).tolist()
        id2embedding[3] = np.random.random(int(meta_info[1])).tolist()

        emb_mat = [id2embedding[idx] for idx in range(len(id2embedding))]
        
        return torch.tensor(emb_mat, dtype=torch.float), token2id, len(vocab)+4


def tokenizer(sent, token2id):
    ids = [token2id.get(token, 1) for token in jieba.cut(sent)]
    return ids


def read_data(config, token2id, mode='train'):
    data_df = pd.read_csv(config[f'{mode}_file_path'], sep=',')
    if mode == 'train':
        X_train, y_train = defaultdict(list), []
        X_val, y_val = defaultdict(list), []
        num_val = int(config['train_val_ratio'] * len(data_df))
    
    else:
        X_test, y_test = defaultdict(list), []

    for i, row in tqdm(data_df.iterrows(), desc=f'Preprocesing {mode} data', total=len(data_df)):
        label=row[1] if mode == 'train' else 0
        sentence = row[-1]
        inputs = tokenizer(sentence, token2id)
        if mode == 'train':
            if i < num_val:
                X_val['input_ids'].append(inputs)
                y_val.append(label)
            else:
                X_train['input_ids'].append(inputs)
                y_train.append(label)
        
        else:
            X_test['input_ids'].append(inputs)
            y_test.append(label)

    if mode == 'train':
        label2id = {label: i for i, label in enumerate(np.unique(y_train))}
        id2label = {i: label for label, i in label2id.items()}

        y_train = torch.tensor([label2id[label] for label in y_train], dtype=torch.long)
        y_val = torch.tensor([label2id[label] for label in y_val], dtype=torch.long)

        return X_train, y_train, X_val, y_val, label2id, id2label
    
    else:
        y_test = torch.tensor(y_test, dtype=torch.long)
        return X_test, y_test

class TNEWSDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __getitem__(self, idx):
        return {
            'input_ids': self.x['input_ids'][idx],
            'label': self.y[idx]
        }
    
    def __len__(self):
        return self.y.size(0)


def collate_fn(examples):
    input_ids_list =[]
    labels = []
    for example in examples:
        input_ids_list.append(example['input_ids'])
        labels.append(example['label'])
    
    # find the max length
    max_length = max(len(input_ids) for input_ids in input_ids_list)

    input_ids_tensor = torch.zeros((len(labels), max_length), dtype=torch.long)

    for i, input_ids in enumerate(input_ids_list):
        seq_len = len(input_ids)
        input_ids_tensor[i, :seq_len] = torch.tensor(input_ids, dtype=torch.long)

    return {
        'input_ids': input_ids_tensor,
        'label': torch.tensor(labels, dtype=torch.long)
    }

from torch.utils.data import DataLoader
def build_dataloader(config, token2id):
    X_train, y_train, X_val, y_val, label2id, id2label = read_data(config, token2id, mode='train')
    X_test, y_test = read_data(config, token2id, mode='test')

    train_dataset = TNEWSDataset(X_train, y_train)
    val_dataset = TNEWSDataset(X_val, y_val)
    test_dataset = TNEWSDataset(X_test, y_test)
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=False, collate_fn=collate_fn)

    return label2id, id2label, train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    pass
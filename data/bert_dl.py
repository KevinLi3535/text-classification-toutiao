

import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import  tqdm
from transformers import BertTokenizer

def read_data(config, tokenizer, mode='train'):
    data_df = pd.read_csv(config[f'{mode}_file_path'], sep=',')
    if mode == 'train':
        X_train, y_train = defaultdict(list), []
        X_val, y_val = defaultdict(list), []
        num_val = int(len(data_df) * config['train_val_ratio'])
    else:
        X_test, y_test = defaultdict(list), []

    for i, row in tqdm(data_df.iterrows(), desc=f'preprocess {mode} data', total=len(data_df)):
        label=row[1] if mode == 'train' else 0
        sentence = row[-1]
        inputs = tokenizer.encode_plus(sentence,
                                      add_special_tokens=True,
                                      return_token_type_ids=True,
                                      return_attention_mask=True)

        if mode == 'train':
            if i < num_val:
                X_val['inputs_ids'].append(inputs['input_ids'])
                y_val.append(label)
                X_val['token_type_ids'].append(inputs['token_type_ids'])
                X_val['attention_mask'].append(inputs['attention_mask'])
                                
            else:
                X_train['inputs_ids'].append(inputs['input_ids'])
                y_train.append(label)
                X_train['token_type_ids'].append(inputs['token_type_ids'])
                X_train['attention_mask'].append(inputs['attention_mask'])

        else:
            X_test['inputs_ids'].append(inputs['input_ids'])
            y_test.append(label) 
            X_test['token_type_ids'].append(inputs['token_type_ids'])
            X_test['attention_mask'].append(inputs['attention_mask'])
            

    if mode == 'train':
        label2id = {label: i for i, label in enumerate(np.unique(y_train))} 
        id2label = {i: label for label, i in label2id.items()} 
        y_train = torch.tensor([label2id[i] for i in y_train], dtype=torch.long)  
        y_val = torch.tensor([label2id[i] for i in y_val], dtype=torch.long)  
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
            'inputs_ids' : self.x['inputs_ids'][idx],
            'label' : self.y[idx],
            'token_type_ids':self.x['token_type_ids'][idx],
            'attention_mask':self.x['attention_mask'][idx],
        }
    
    def __len__(self):
        return self.y.size(0)


def collate_fn(examples):
    input_ids_list = []
    labels = []
    token_type_ids_list = []
    attention_mask_list = []

    for example in examples:
        input_ids_list.append(example['inputs_ids'])
        labels.append(example['label'])

        token_type_ids_list.append(example['token_type_ids'])
        attention_mask_list.append(example['attention_mask'])
    
    max_length = max(len(input_ids) for input_ids in input_ids_list)
    input_ids_tensor = torch.zeros((len(labels), max_length), dtype=torch.long)

    token_type_ids_tensor = torch.zeros_like(input_ids_tensor)
    attention_mask_tensor = torch.zeros_like(input_ids_tensor)

    for i, input_ids in enumerate(input_ids_list):
        input_ids_tensor[i, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids_tensor[i, :len(token_type_ids_list[i])] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
        attention_mask_tensor[i, :len(attention_mask_list[i])] = torch.tensor(attention_mask_list[i], dtype=torch.long)
        
    
    return{
        'input_ids' : input_ids_tensor,
        'labels' : torch.tensor(labels, dtype=torch.long),
        'token_type_ids': token_type_ids_tensor,
        'attention_mask': attention_mask_tensor
    }

from transformers import BertTokenizer
from torch.utils.data import DataLoader
def build_dataloader(config):

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['model_path'])
    X_train, y_train, X_val, y_val, label2id, id2label = read_data(config, tokenizer, mode='train')
    X_test, y_test = read_data(config, tokenizer, mode='test')

    train_dataset = TNEWSDataset(X_train, y_train)
    val_dataset = TNEWSDataset(X_val, y_val)
    test_dataset = TNEWSDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=2, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, id2label

if __name__ == "__main__":
    pass
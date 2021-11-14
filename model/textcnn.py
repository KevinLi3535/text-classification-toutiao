"""
TextCNN Model for Text Classifications

Reference: Convolutional Neural Networks for Sentence Classification
           https://arxiv.org/abs/1408.5882
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm,trange
from sklearn.metrics import f1_score
import pandas as pd
import os

# Model Class
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(config['embedding_pretrained'], freeze=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, config['num_filters'], (k, config['emb_size'])) for k in config['filter_sizes']])
        self.dropout = nn.Dropout(config['dropout'])
        self.fc = nn.Linear(len(config['filter_sizes'] * config['num_filters']), config['num_classes'])

    def convs_and_pool(self, x, conv):

        # x [batch_size, out_channels, seq_len_out, 1]
        # x [batch_size, out_channels, seq_len_out]
        x = F.relu(conv(x)).squeeze(3)

        # x (batch_size, out_channels, 1)
        # x (batch_size, out_channels)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids=None, label=None):
        # out [batch_size, seq_len, embedding_dim]
        out = self.embedding(input_ids)
        
        # H: seq_len; W:embedding_dim
        # out [batch_size, 1, seq_len, embedding_dim]
        out = out.unsqueeze(1)

        # (batch_size, out_channels)
        out = torch.cat([self.convs_and_pool(out, conv) for conv in self.convs], 1)

        out = self.dropout(out)

        out = self.fc(out)

        output = (out, )

        if label is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(out, label)
            output = (loss, ) + output

        return output

# Training Loop
def train(model, config, id2label, train_dataloader, val_dataloader):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    model.to(config['device'])
    epoches_iterator = trange(config['num_epochs'])
    
    global_step = 0
    train_loss = 0.
    logging_loss = 0.

    for epoch in epoches_iterator:
        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch = {item: value.to(config['device']) for item, value in batch.items()}

            loss = model(**batch)[0]

            model.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss
            global_step += 1

            if global_step % config['logging_step'] == 0:
                print_train_loss = (train_loss - logging_loss) / config['logging_step']
                logging_loss = train_loss

                avg_val_loss, f1 = evaluation(config, model, val_dataloader)
                print_log = f'>>> training loss: {print_train_loss:.4f}, valid loss: {avg_val_loss:.4f}, valid f1 score: {f1:.4f}'
                print(print_log)
                model.train()

    return model

def evaluation(config, model, val_dataloader):
    model.eval()
    preds = []
    labels = []
    val_loss =0.
    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))
    with torch.no_grad():
        for batch in val_iterator:
            labels.append(batch['label'])
            batch = {item: value.to(config['device']) for item, value in batch.items()}

            # val output (loss, out)
            loss, logits = model(**batch)[:2]
            val_loss += loss.item()

            preds.append(logits.argmax(dim=-1).detach().cpu())
    
    avg_val_loss = val_loss/len(val_dataloader)
    labels = torch.cat(labels, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()

    f1 = f1_score(labels, preds, average='macro')
    return avg_val_loss, f1


def predict(config, id2label, model, test_dataloader):
    test_iterator = tqdm(test_dataloader, desc='Evaluation', total=len(test_dataloader))
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_iterator:
            batch = {item: value.to(config['device']) for item, value in batch.items()}

            logits = model(**batch)[1]

            test_preds.append(logits.argmax(dim=-1).detach().cpu())
    
    test_preds = torch.cat(test_preds, dim=0).numpy()
    test_preds = [id2label[id_] for id_ in test_preds]

    test_df = pd.read_csv(config['test_file_path'], sep=',')
    # test_df.insert(1, column='label', value=test_preds)
    test_df.drop(columns=['sentence'], inplace=True)
    test_df.to_csv(os.path.join(config['project_dir'],'submission.csv'), index=False, encoding='utf8')
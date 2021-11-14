"""
Bert Model (Sequence Classification)

Reference: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
           https://arxiv.org/abs/1810.04805
"""

from transformers import BertForSequenceClassification,BertConfig,AdamW
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm,trange
from sklearn.metrics import f1_score
import pandas as pd
import os

# Model Class
## for this "pure" Bert Model, we don't neccessarily need 

def train(config, id2label, train_dataloader, val_dataloader):

    bert_config = BertConfig.from_pretrained(config['model_path'])
    bert_config.num_labels = len(id2label)
    model = BertForSequenceClassification.from_pretrained(config['model_path'],config=bert_config)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    model.to(config['device'])
    epoch_iterator = trange(config['num_epochs'])
    global_steps = 0
    train_loss = 0.
    logging_loss = 0.

    for epoch in epoch_iterator:

        train_iterator = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))
        model.train()
        for batch in train_iterator:
            batch = {item: value.to(config['device']) for item, value in batch.items()}

            loss = model(**batch)[0]

            model.zero_grad()

            loss.backward()
            
            optimizer.step()

            train_loss += loss.item()
            global_steps += 1

            if global_steps % config['logging_step'] == 0:
                print_train_loss = (train_loss - logging_loss) / config['logging_step']
                logging_loss = train_loss

                avg_val_loss, f1 = evaluation(config, model, val_dataloader)

                print_log = f'>>> training loss: {print_train_loss:.4f}, valid loss: {avg_val_loss:.4f}, ' \
                            f'valid f1 score: {f1:.4f}'
                print(print_log)
                model.train()

    return model


def evaluation(config, model, val_dataloader):
    model.eval()
    preds = []
    labels = []
    val_loss = 0.
    val_iterator = tqdm(val_dataloader, desc='Evaluation', total=len(val_dataloader))

    with torch.no_grad():
        for batch in val_iterator:
            labels.append(batch['labels'])
            batch = {item: value.to(config['device']) for item, value in batch.items()}
            loss, logits = model(**batch)[:2]

            val_loss += loss.item()
            preds.append(logits.argmax(dim=-1).detach().cpu())

    avg_val_loss = val_loss / len(val_dataloader)
    labels = torch.cat(labels, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    f1 = f1_score(labels, preds, average='macro')
    return avg_val_loss, f1

## prediction using test data
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
    test_df.insert(1, column='label', value=test_preds)
    test_df.drop(columns=['sentence'], inplace=True)
    test_df.to_csv(os.path.join(config['project_dir'],'submission.csv'), index=False, encoding='utf8')
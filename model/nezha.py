"""
NeZha

Reference: NEZHA: Neural Contextualized Representation for Chinese Language Understanding
           https://arxiv.org/abs/1909.00204
"""

from transformers import AdamW
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm,trange
from sklearn.metrics import f1_score
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model_config.nezha_config.NeZha import *

## Model Class
class NeZhaForTNEWS(NeZhaPreTrainedModel):
    def __init__(self,config,model_path, classifier):
      super(NeZhaForTNEWS,self).__init__(config)
      self.bert = NeZhaModel.from_pretrained(model_path,config=config)
      self.classifier = classifier

    def forward(self,input_ids,token_type_ids,attention_mask,labels):
      outputs = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)  # NeZha model does not have hidden states
      
      ## instead of using last hidden state, let's take out all hidden states
      ## In the future, we may be able to build models, such as LSTM, RNN
      hidden_states = outputs[2]

      ## input hidden states to head
      logits = self.classifier(hidden_states,input_ids)
      outputs = (logits, )

      if labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits,labels.view(-1))
        outputs = (loss,)+outputs

      return outputs

## Head Classes
class ConvClassifier(nn.Module):
  "CNN + Global Max Pool"
  def __init__(self,config):
    super().__init__()
    self.conv = torch.nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size,kernel_size=3)
    self.global_max_pool = torch.nn.AdaptiveMaxPool1d(1)
    self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
    self.fc = torch.nn.Linear(config.hidden_size,config.num_labels)

  def forward(self,hidden_states,input_ids):
    ## only take out the last step
    hidden_states = self.dropout(hidden_states[-1])
    ## (batch_size, seq_len, hidden_size) --> (batch_size, hidden_size, seq_len)
    hidden_states = hidden_states.permute(0,2,1)
    ## (batch_size, hidden_size, seq_len_out)
    out = F.relu(self.conv(hidden_states))
    
    out = self.global_max_pool(out).squeeze(dim=2)
    out = self.fc(out)

    return out

## NeZha + Head
def build_model(model_path, config, head):
  heads={
      'cnn':ConvClassifier
  }

  model = NeZhaForTNEWS(config,model_path,heads[head](config))
  return model

# NeZha_model + head train
def train(config, id2label, train_dataloader, val_dataloader):

    bert_config = NeZhaConfig.from_pretrained(config['model_path'])
    bert_config.output_hidden_states = True
    bert_config.num_labels = len(id2label)

    model = build_model(config['model_path'],bert_config,config['head'])
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
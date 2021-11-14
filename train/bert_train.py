import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.config import *
from model.bert import *
from data.bert_dl import *


if __name__ == "__main__":
    
    config.update(
        {
        'batch_size': 16,
        'num_epochs': 1,
        'learning_rate': 2e-5,
        'logging_step': 500,
        'head':'cnn',
        'model_path':os.path.join(config['model_config_dir'],'bert_config')
        }
    )
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # cpu or gpu

    seed_everything(config['seed'])

    train_dataloader, val_dataloader, test_dataloader, id2label = build_dataloader(config)
    # Bert Training
    model = train(config, id2label, train_dataloader, val_dataloader)
    print('######### Training Bert for Sequence Ended ##########')
    evaluation(config, model,val_dataloader)
    torch.save(model.state_dict(), os.path.join(config['project_dir'],'bert.pt'))
    predict(config, id2label, model, test_dataloader)
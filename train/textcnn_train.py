
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train.config import *
from model.textcnn import *
from data.textcnn_dl import *

if __name__ == "__main__":
    
    config.update(
        {
        'vocab_size': 30000,   # dictionary size
        'batch_size': 64,
        'num_epochs': 1,
        'learning_rate': 1e-3,
        'logging_step': 300,
        }
    )
    
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # cpu or gpu

    seed_everything(config['seed'])

    with bz2.open(os.path.join(config['model_config_dir'],'textcnn_config','sgns.weibo.word.bz2')) as f:
        token_vector = f.readlines()

    vocab = get_vocab(config)
    emb_mat, token2id, config['vocab_size'] = get_embedding(vocab)
    label2id, id2label, train_dataloader, val_dataloader, test_dataloader = build_dataloader(config, token2id)

    model_config = {
        'embedding_pretrained' : emb_mat,
        'num_filters' : 256,
        'emb_size' : emb_mat.shape[1],
        'dropout' : 0.3,
        'filter_sizes' : [2,3,5],
        'num_classes' : len(label2id)
    }

    model = Model(model_config)
    model = train(model, config, id2label, train_dataloader, val_dataloader)
    
    print('######### Training Text CNN Ended ##########')
    evaluation(config, model,val_dataloader)
    torch.save(model.state_dict(), os.path.join(config['project_dir'],'textcnn.pt'))
    predict(config, id2label, model, test_dataloader)
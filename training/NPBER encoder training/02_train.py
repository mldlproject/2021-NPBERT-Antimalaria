# Import Libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import pytorch_lightning as pl
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    DataCollatorForLanguageModeling
)
from model import Bert
from torch.utils.data import DataLoader, Dataset

#==================================================
# Set Paths
tokenizer_vocab_path = '/content/drive/MyDrive/Transformer/data_processed/tokenizer.txt'
model_save_dir = '/content/drive/MyDrive/Transformer/save_model/'
dataset = '/content/drive/MyDrive/Transformer/data_processed/dataset.npy'

#==================================================
# Set Config Parameters
use_gpu = torch.cuda.is_available()
NHEADS = 4
VOCAB_SIZE = 24542 # có thể điều chỉnh
HIDDEN_SIZE = 128 # 64, 128, 256
HIDDEN_LAYERS = 5 # 3,4,5
INTERMEDIATE_SIZE = 256 # 64, 128, 256
MAX_POSITION_EMBEDDINGS = 512
random_seed = 0
NUM_EPOCH = 10 
BATCH_SIZE = 32
random_seed = 0

# Tokenizer
tokenizer = BertTokenizer(tokenizer_vocab_path)

#==================================================
# Set Model Parameters 
config = BertConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE*NHEADS,
    num_hidden_layers=HIDDEN_LAYERS,
    num_attention_heads=NHEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    output_hidden_states = True
 
)

#==================================================
# Loading and Split Dataset
dataset = np.load(dataset, allow_pickle=True)
X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=42) 

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

train_loader = DataLoader(
    X_train,
    batch_size= 32,
    shuffle = True,
    collate_fn=data_collator   
)
eval_loader = DataLoader(
    X_test,
    batch_size= 32,
    shuffle = False,
    collate_fn=data_collator   
)

#==================================================
# Initialise model
model = Bert(use_gpu, config)
# model.load_state_dict(torch.load('/content/drive/MyDrive/Transformer Learning/save_model/saved_model_epoch20.bin'))
optimizer = model.configure_optimizers()
def train(epoch):
    model.train()
    train_loss = 0
    for idx , batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model.training_step(batch, idx)

        loss = output.loss
        train_loss += loss.detach().cpu()  
        
        loss.backward()
        
        if idx % 1000 == 0 :
            print('Training loss of batch_{}/epoch{} = {}'.format(idx, epoch, loss))
        optimizer.step()
    train_loss = train_loss/len(train_loader)
    return output, train_loss

def val(epoch):
    model.eval()
    val_loss = 0
    for idx , batch in enumerate(eval_loader):
        output = model.training_step(batch, idx)
        loss = output.loss
        val_loss += loss.detach().cpu()   
    val_loss = val_loss/len(eval_loader) 
    print('Validation loss of epoch{} = {}'.format(epoch, val_loss))  
    return output, val_loss

training_loss = []
validation_loss = []
val_check = 100
for epoch in range(1, 11):

    output, train_loss = train(epoch)
    output, val_loss = val(epoch)
    # Save loss
    training_loss.append(train_loss)
    validation_loss.append(val_loss)
    if val_loss < val_check: 
        # save model best model
        torch.save(model.state_dict(), model_save_dir + 'pytorch_model.bin')
        val_check = val_loss

#==================================================
# Save Model Loss 
np.save('/content/drive/MyDrive/Transformer/loss/training_loss.npy', training_loss)
np.save('/content/drive/MyDrive/Transformer/loss/validation_loss.npy', validation_loss)  

# Import Libraries
import pandas as pd
import numpy as np
import pickle
import codecs, json
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
from transformers import LineByLineTextDataset

#==================================================
# Define Params
data_path = '/content/drive/MyDrive/Transformer/data_processed/data_preprocess.txt'
tokenizer_output_path = '/content/drive/MyDrive/Transformer/data_processed/tokenizer.json'
tokenizer_vocab_path = '/content/drive/MyDrive/Transformer/data_processed/tokenizer.txt'
MAX_POSITION_EMBEDDINGS = 520

#==================================================
# Train and Save Tokenizer
tokenizer = BertWordPieceTokenizer(clean_text=True, lowercase=False)
tokenizer.train(data_path, vocab_size=20000, min_frequency=0, show_progress=True)
tokenizer.save(tokenizer_output_path)
print('Tokenizer trained')

vocab = set()
with codecs.open(data_path,'r','utf-8') as f:
    data = f.readlines()
    # print(len(data))
for line in data:
    for d in line.strip().split():
        # print(d)
        vocab.add(d)
# print(len(vocab))

with codecs.open(tokenizer_output_path,'r') as f:

    tokenizerModel = json.loads(f.read())
    vocab1 = set(list(tokenizerModel['model']['vocab'])+list(vocab))
    with codecs.open(tokenizer_vocab_path,'w','utf-8') as vocab_f:
        vocab_f.write('\n'.join([v for v in vocab1 if '#' not in v]))

tokenizer = BertTokenizer(tokenizer_vocab_path)  

#==================================================
# Build Training Set
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=data_path,
    block_size=MAX_POSITION_EMBEDDINGS
)

# Save Dataset 
np.save('/content/drive/MyDrive/Transformer/data_processed/dataset.npy', dataset)

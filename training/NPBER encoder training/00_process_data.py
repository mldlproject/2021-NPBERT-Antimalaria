# Import Libraries
import pandas as pd
import numpy as np
from utils import read_data, generate_Bert_corpus

#==================================================
# Generate Corpus 
print('processing data')
PATH = '/content/drive/MyDrive/Transformer/data'
df = read_data(PATH)

OUTPUT = '/content/drive/MyDrive/Transformer/data_processed/data_preprocess.txt'

generate_Bert_corpus(
    pdseries = df,
    output_dir = OUTPUT
)

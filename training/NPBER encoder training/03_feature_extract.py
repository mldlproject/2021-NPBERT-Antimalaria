# Import Libraries
import pandas as pd
import numpy as np
import pickle
import codecs, json
from rdkit import Chem
from transformers import BertTokenizer
from utils import gen_sentence, df2features

#==================================================
# Set Paths
INPUTDIR = '/content/drive/MyDrive/Transformer/data_clean/antimalarial_comp_list.csv' # this file contains the anti-malaria smiles dataset
MODEL_DIR = '/content/drive/MyDrive/Transformer/save_model'
tokenizer_vocab_path = '/content/drive/MyDrive/Transformer/data_processed/tokenizer.txt'
output_feature = '/content/drive/MyDrive/Transformer/output/feature.csv'

#==================================================
def generate_df(dir,sep='\t',header=None,names='smiles id y'.split()):
    df = pd.read_csv(dir,sep=sep,header=header,names=names)
    df['sentence'] = df.smiles.apply(lambda x: gen_sentence(x))
    return df

df = generate_df(INPUTDIR)
print(df)

#==================================================
from transformers import BertTokenizer
import transformers
from transformers import pipeline

btokenizer = BertTokenizer(tokenizer_vocab_path)
extract_feature = pipeline(
    "feature-extraction",
    model=MODEL_DIR,
    tokenizer=btokenizer
)

x, y = df2features(df, extract_feature, tgt_pos_label='A')
dfx = pd.DataFrame(x).rename(columns={0:'Smiles'})
# print(dfx)
dfx.to_csv(output_feature)
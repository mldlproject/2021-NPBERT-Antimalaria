# Library
import pandas as pd
import numpy as np
import pickle
import codecs, json
from rdkit import Chem
from transformers import BertTokenizer
from utils import gen_sentence, df2features, generate_feature
import argparse
import transformers
from transformers import pipeline

#############################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--input_smile', type=str, default= None, help='Check feature of smile')
parser.add_argument('--input_file', type=str, help='Location for the input dataset')
parser.add_argument('--model_dir', type=str, default='./save_model', help='Location for save pretrain model')
parser.add_argument('--tokenizer_vocab_path', type=str, default='./save_model/tokenizer.txt', help='Location for tokenizer vocab')
parser.add_argument('--output_file', type=str, help='Location for the dataset')

args = parser.parse_args()

#############################################################################################
# File directories
INPUTDIR = args.input_file # this file contains the anti-malaria smiles dataset
MODEL_DIR = args.model_dir
tokenizer_vocab_path = args.tokenizer_vocab_path
output_feature = args.output_file

def generate_df(dir,sep='\t',header=None,names='smiles id y'.split()):
    df = pd.read_csv(dir,sep=sep,header=header,names=names)
    df['sentence'] = df.smiles.apply(lambda x: gen_sentence(x)) # lambda x: gen_sentence(x) : Thực hiện hàm gen_sentence 
    return df

if __name__ == '__main__':
    btokenizer = BertTokenizer(tokenizer_vocab_path)
    extract_feature = pipeline(
            "feature-extraction",
            model=MODEL_DIR,
            tokenizer=btokenizer
        )

    if args.input_smile is not None:
        print('Smile: ', args.input_smile)
        feature = generate_feature(args.input_smile, extract_feature)
        if feature is not None:
            print('shape of feature ',feature.shape)
            print(feature)
    else:
        df = generate_df(INPUTDIR)
        print(df)
        x, y = extract_features(df, extract_feature, tgt_pos_label='A')
        dfx = pd.DataFrame(x).rename(columns={0:'Smiles'})
        dfx.to_csv(output_feature)
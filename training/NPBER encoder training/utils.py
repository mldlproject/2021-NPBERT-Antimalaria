# Import libraries
import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
import gc
gc.enable()
from rdkit import Chem as chem
from rdkit.Chem import AllChem
import pickle
import codecs,json
from gensim.models import word2vec
from transformers import pipeline

#==================================================
# Read input data
def read_data(dir):
    # Select files in fold then put in list
    files = [(join(dir,f),'-'.join(f.split('.')[0].split('-')[1:])) for f in listdir(dir) if (f.split('.')[-1] == 'smiles' or f.split('.')[-1] == "csv")]
    df = []
    for f in files:
        if f[0].split('.')[-1] == 'smiles':
            df_ = pd.read_csv(f[0], sep='\t', header=None, names='smiles id y'.split())
            df.append(df_)
        
        if f[0].split('.')[-1] == 'csv':
            df_ = pd.read_csv(f[0])
            df.append(df_)

    df = pd.concat(df, ignore_index=True)  
    res = df.smiles.dropna().unique() # Delete dupplicated SMILES
    print(f'Generated corpus of length {len(res)}')
    return res

#==================================================
# Generate and save the corpus
def generate_Bert_corpus(pdseries, output_dir, include_smiles = False, max_length=512):
    corpus = []
    sent_max_length = 0
    for i in range(len(pdseries)):
        smiles = pdseries[i]
        sentence = gen_sentence(smiles, max_length)
        if sentence is None:
            continue
        sentence = ' '.join(sentence)
        if include_smiles:
            sentence = smiles + ' ' + sentence
        corpus.append(sentence)
        
    print('Corpus Length: {:0,.0f}'.format(len(corpus)))
    corpus = '\n'.join(corpus)
    print('Corpus max len: {:0,.0f}'.format(sent_max_length))
    if output_dir is None:
        return corpus
    else:
        with codecs.open(output_dir,'w','utf-8') as f:
            f.write(corpus)

#==================================================
# Create a sentence from SMILES
def gen_sentence(smiles, max_length=2032):# max_length = 2032
    try:
        mol = chem.MolFromSmiles(smiles) # Convert SMIELS to mol 
    except:
        return None
    if mol is None:
        return None
    numAtoms = mol.GetNumAtoms() # Number of atoms
    if numAtoms*2 > max_length:
        return None
    sentence = ['[UNK]']*numAtoms*2
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, 1, bitInfo=info)
    for key, tup in info.items():
        for t in tup:
            atom_index = t[0]
            r = t[1]
            k = str(key)                    
            sentence[2*atom_index+r] = k
    return sentence

#==================================================
# Create NNPERT-encoded Features 
def df2features(df, pipeline, sentence = 'sentence', tgt = 'y', tgt_pos_label=1):
    df2 = df[~pd.isna(df[sentence])]
    x = []
    p20 = int(len(df)/20)
    print('pipelining {:0,.0f} sentences ['.format(len(df)), end='')
    # np.asarray([df.iloc[i].smiles] --> Chuoi smiles 
    # pipeline(' '.join(df.iloc[i][sentence]))[0][0] --> feature of smiles 
    for i in range(len(df)):
        x.append(np.asarray([df.iloc[i].smiles] + pipeline(' '.join(df.iloc[i][sentence]))[0][0]))
    print('] Done.')
    print(np.asarray(x).shape)
    return np.asarray(x), (df2[tgt]==tgt_pos_label).astype(float)
"""Functions for reading in the NEWTS dataset, along with the corresponding LDA
model (if desired). 
"""
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

# returns a Pandas DataFrame of the NEWTS testing set
def read_test(path="NEWTS_test_600.csv"):
    out = pd.read_csv(path, encoding='utf-8', index_col=[0])
    assert(len(out) == 600)
    return out

# returns a Pandas DataFrame of the NEWTS training set
def read_train(path="NEWTS_train_2400.csv"):
    out = pd.read_csv(path, encoding='utf-8', index_col=[0])
    assert(len(out) == 2400)
    return out

# given a path to model, loads it and its dictionary
def read_lda(path="250/"):
    path = path.strip('/') + '/'
    lda = LdaModel.load(path+'lda.model', mmap = 'r')
    print("Loaded the LDA model")
    dictionary = Dictionary.load(path+'dictionary.dic', mmap = 'r')
    print("Loaded dictionary")
    return lda, dictionary

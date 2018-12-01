import pandas as pd
import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.cluster import SpectralClustering


class CandidateStore:
    def __init__(self, n_clusters):
        self.docs = pd.DataFrame(columns=['text','cluster','topic'])
        self.dictionary = None
        self.term_mat = None
        self.n_clusters = clusters


    def retrieve(self, question):
        pass

    def add_doc(self, texttxt):
        f = open(texttxt,'r')
        paragraph = ""
        for l in f:
            print(l)
            if l == "\n":
                self.docs.loc[len(self.docs)] = [paragraph, np.nan, None]
                paragraph = ""
            else:
                paragraph += l[:-1]


    def get_num_candidates(self):
        return self.docs.shape[0]

    def get_all_candidates(self):
        return self.docs

    def save_store(self):
        pass

    def make_clusters(self):
        self.dictionary, self.term_mat = make_token_column(self.docs)
        self.clusters = SpectralClustering(n_clusters,
                assign_labels="discretize").fit(self.term_mat)
        self.docs["cluster"] = self.clusters.labels_

def tokenize_wo_stops(doc):
    return [w for w in word_tokenize(doc) if w not in stopwords.words("english")]

def make_token_column(df, remove_stopwords=True):
    trans = str.maketrans("", "", string.punctuation)
    pattern = r"http\S+|"
    df["edit"] = df["text"].str.replace(pattern, "").str.lower().str.translate(trans)
    
    if remove_stopwords:
        df["tokenized"] = df["edit"].apply(tokenize_wo_stops)
    else:
        df["tokenized"] = df["edit"].apply(word_tokenize)
    
    dct = Dictionary(df["tokenized"])
    
    df["bow"] = df["tokenized"].apply(dct.doc2bow)
    
    dct_term = corpus2csc(df["bow"]).todense().T
    
    return dct, dct_term

if __name__ == '__main__':
    CS = CandidateStore()
    # CS.add_doc('on_method.txt')
    # print(CS.get_num_candidates())
    # print(CS.get_all_candidates())


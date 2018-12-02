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
        self.n_clusters = n_clusters

    def retrieve(self, question,remove_stopwords=True):
        q = pd.DataFrame(columns=['text','cluster','topic'])
        q.loc[len(q)] = [question,np.nan,None]
        trans = str.maketrans("", "", string.punctuation)
        pattern = r"http\S+|"
        q['edit'] = q['text'].str.replace(pattern, "").str.lower().str.translate(trans)
        if remove_stopwords:
            q['tokenized'] = q['edit'].apply(tokenize_wo_stops)
        else:
            q['tokenized'] = q['edit'].apply(word_tokenize)
        q['bow'] = q['tokenized'].apply(self.dictionary.doc2bow)
        dct_term = corpus2csc(q['bow'],num_terms=self.term_mat.shape[1]).todense().T
        distance = np.linalg.norm(self.term_mat - dct_term,axis=1)
        cluster_num = self.docs['cluster'][np.argmin(distance)]
        return self.docs.loc[self.docs['cluster'] == cluster_num]['text']

    def add_doc(self, texttxt):
        f = open(texttxt,'r')
        paragraph = ""
        for l in f:
            if l == "\n":
                self.docs.loc[len(self.docs)] = [paragraph, np.nan, None]
                paragraph = ""
            else:
                paragraph += l[:-1] + " "


    def get_num_candidates(self):
        return self.docs.shape[0]

    def get_all_candidates(self):
        return self.docs

    def save_store(self):
        pass

    def make_clusters(self):
        self.dictionary, self.term_mat = make_token_column(self.docs)
        # print(self.dictionary,self.term_mat,self.term_mat.shape)
        self.clusters = SpectralClustering(self.n_clusters,
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
    CS = CandidateStore(10)
    CS.add_doc('on_method.txt')
    CS.make_clusters()
    print(CS.retrieve("What is coordinate system?"))
    print(CS.get_num_candidates())
    # print(CS.get_all_candidates())

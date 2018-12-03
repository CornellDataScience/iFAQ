import datetime
import pandas as pd
import numpy as np
import string
import os

from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.cluster import SpectralClustering


class CandidateStore:
    def __init__(self, n_clusters):
        self.docs = pd.DataFrame(columns=['text', 'cluster', 'topic', 'edit', 'tokenized', 'bow'])
        self.dictionary = None
        self.term_mat = None
        self.n_clusters = n_clusters
        self.file_info = []

    def retrieve(self, question, remove_stopwords=True):
        q = pd.DataFrame(columns=['text', 'cluster', 'topic', 'edit', 'tokenized', 'bow'])
        q.loc[len(q)] = [question, np.nan, None, None, None, None]
        trans = str.maketrans("", "", string.punctuation)
        pattern = r"http\S+|"
        q['edit'] = q['text'].str.replace(pattern, "").str.lower().str.translate(trans)
        if remove_stopwords:
            q['tokenized'] = q['edit'].apply(tokenize_wo_stops)
        else:
            q['tokenized'] = q['edit'].apply(word_tokenize)
        q['bow'] = q['tokenized'].apply(self.dictionary.doc2bow)
        dct_term = corpus2csc(q['bow'], num_terms=self.term_mat.shape[1]).todense()
        distance = np.matmul(self.term_mat.A,dct_term.A).flatten()
        cluster_num = self.docs['cluster'][np.argmax(distance)]
        return self.docs.loc[self.docs['cluster'] == cluster_num]['text']

    def add_doc(self, texttxt):
        texttxt = os.path.join('app/static/docs/', texttxt)
        self.file_info.append({
            'name': texttxt.split('/')[-1],
            'size': '{}kB'.format(os.path.getsize(texttxt) >> 10),
            'date': datetime.date.today().strftime('%B %d, %Y')
        })
        f = open(texttxt, 'r')
        paragraph = ""
        for l in f:
            if l == "\n":
                self.docs = self.docs.append(
                    pd.DataFrame([[paragraph, np.nan, None, None, None, None]],
                                 columns=self.docs.columns),
                    ignore_index=True)
                paragraph = ""
            else:
                paragraph += l[:-1] + " "
        self.make_clusters()

    def get_num_candidates(self):
        return self.docs.shape[0]

    def get_all_candidates(self):
        return self.docs

    def save_store(self):
        pass

    def make_clusters(self):
        self.dictionary, self.term_mat = make_token_column(self.docs)
        # print(self.dictionary,self.term_mat,self.term_mat.shape)
        self.clusters = SpectralClustering(self.n_clusters, assign_labels="discretize").fit(self.term_mat)
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

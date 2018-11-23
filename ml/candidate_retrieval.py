import pandas as pd
import numpy as np


class CandidateStore:
    def __init__(self):
        self.docs = pd.DataFrame(columns=['text','cluster','topic'])


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
        pass

    def get_all_candidates(self):
        pass

    def save_store(self):
        pass

    def get_clusters(self):
        pass

    def make_clusters(self):
        pass


if __name__ == '__main__':
    CS = CandidateStore()

+import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

from nltk.corpus import stopwords
import nltk

import random
import re

import pickle
import time
import os
import shutil


class embd():
    
    def __init__(self):
        self.embedding_size = 500
    
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')

        self.stop_words = stopwords.words('english')
        self.model1 = Word2Vec.load("word2vec_small.model")

    def avg_sentence_vector(self, words, model, num_features):
        featureVec = np.zeros((1,num_features), dtype="float32")
        count = 0
        for word in words:
            if word in model.wv:
                count += 1
                featureVec = np.append(featureVec, model.wv[word].reshape(1, -1), axis = 0)

        if count > 0:
            featureVec = np.mean(featureVec[1:], axis = 0)
        else:
            return np.zeros((num_features), dtype="float32")
        return featureVec

    def embedding(self, sentence):

        view = re.sub(r'[^\w\s]','', sentence).lower().split()
        view = [word for word in view if word not in self.stop_words]
        ret = self.avg_sentence_vector(view, self.model1, self.embedding_size)

        return ret
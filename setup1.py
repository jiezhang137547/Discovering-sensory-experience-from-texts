import gensim
from gensim.models import Word2Vec
import itertools
import scipy
from scipy import spatial
# from nltk.tokenize.toktok import ToktokTokenizer
import json

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd

tweets_smell = pd.read_csv("tweets_smell.csv")
#replace nan with empty string
tweets_smell["all_triggers"].fillna("", inplace=True)

model_1 = gensim.models.Word2Vec.load("new_data_glove.model")
words = list(model_1.wv.vocab)

def get_word_embeddings1(df_column, b, n):
    word_embedding_dicts1={}
    for i in range(b,n):
        for key in words:
            if key == df_column['all_triggers'].iloc[i]:
                print(i)
                id = str(df_column['id'].iloc[i])
                dic={key:model_1[key].tolist()}
                dicts={id:dic}
                word_embedding_dicts1.update(dicts)
    with open('word_embeddings_smell_related5.json', 'w') as fp:
        json.dump(word_embedding_dicts1, fp, indent=4)
    return word_embedding_dicts1
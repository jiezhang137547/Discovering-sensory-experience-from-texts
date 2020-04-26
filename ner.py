import pandas as pd
import nltk
import json
import numpy as np
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

from nltk.tokenize.toktok import ToktokTokenizer

nlp = spacy.load('en_core_web_sm')

tweets_smell = pd.read_csv("tweets_smell")

def spacy_large_ner(document):
    entity_dicts={}
    for ent in nlp(document).ents:
        dic = {ent.text.strip(): ent.label_}
        entity_dicts.update(dic)
    return entity_dicts


entities_dicts_all = {}
for i in range(len(tweets_smell['processed_tweets'])):
    key = tweets_smell['id'].iloc[i]
    entities = spacy_large_ner(tweets_smell['processed_tweets'].iloc[i])
    entities_dic = {key: entities}
    entities_dicts_all.update(entities_dic)

entities_key_list = list(entities_dicts_all.keys())

print(entities_dicts_all.get(entities_key_list[0]))
entity_loc_dict={}
for i in range(len(entities_key_list)):
    for key, value in entities_dicts_all.get(entities_key_list[i]).items():
        if value == 'LOC':
            dic = {key:entities_dicts_all.get(entities_key_list[i]).get(key)}
            dics={entities_key_list[i]:dic}
            entity_loc_dict.update(dics)
print(entity_loc_dict)
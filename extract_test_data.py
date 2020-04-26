import pandas as pd
import json
import numpy as np
import add_remove_triggers
import ner


#
#"tweets_smell" 需要用Jupyter notebook里面的NER training 文件
tweets_smell = pd.read_csv("tweets_smell")
tweets_smell["all_triggers"].fillna("", inplace=True)

embeddings_dict = add_remove_triggers.embeddings_dict_reduced

id_list = [int(i) for i in list(embeddings_dict)]

def embedding_pairs(dicts):
    word_embedding_pairs = {}
    for i in range(len(id_list)):
        key = str(id_list[i])
        word_embedding_pair = dicts.get(key)
        word_embedding_pairs.update(word_embedding_pair)
    return word_embedding_pairs

word_embedding_pairs = embedding_pairs(embeddings_dict)

word_embedding_pairs_arrays={}
for k, v in word_embedding_pairs.items():
    word_embedding_pairs_array = np.array(word_embedding_pairs.get(k))
    dicts = {k:word_embedding_pairs_array}
    word_embedding_pairs_arrays.update(dicts)

words = list(word_embedding_pairs)



vectors = [word_embedding_pairs[word] for word in words]


word_list = []
for i in range(len(tweets_smell)):
    if tweets_smell['all_triggers'].iloc[i] in words:
        word_list.append(tweets_smell['all_triggers'].iloc[i])
    else:
        word_list.append('')



embedding_lists=[]
for i in range(len(word_list)):
    if word_list[i] != '':
        embedding_list = word_embedding_pairs.get(word_list[i])
        # print(embedding_list)
        list1 = [j for j in embedding_list]
        embedding_lists.append(list1)
    else:
        embedding_lists.append('')





word_embedding_df = pd.DataFrame(embedding_lists,word_list)


word_embedding_df = word_embedding_df.reset_index()
word_embedding_df['id'] = tweets_smell['id']

word_embedding_df['processed_tweets'] = tweets_smell['processed_tweets']
word_embedding_df['LOC'] = tweets_smell['LOC']
word_embedding_df['place'] = tweets_smell['place']

entity_loc_dict = ner.entity_loc_dict
ids = list(entity_loc_dict)
print(len(ids))


loc_pretrained_list=[]
for i in range(len(ids)):
    for key, value in entity_loc_dict.get(ids[i]).items():
        loc_pretrained = key
        loc_pretrained_list.append(loc_pretrained)

df = {'id':ids, 'loc_pretrained': loc_pretrained_list}
loc_pretrained_df = pd.DataFrame(df)
word_embedding_df = word_embedding_df.merge(loc_pretrained_df, how='outer', on='id')






word_embedding_df = word_embedding_df.replace(r'', np.nan, regex=True)

all_tweets_with_smell_triggers = word_embedding_df[word_embedding_df['index'].notnull()]
print(len(all_tweets_with_smell_triggers))

all_tweets_with_place = word_embedding_df[word_embedding_df['LOC'].notnull()]
print(len(all_tweets_with_place))

#
# for i in range(len(all_tweets_with_smell_triggers)):
#     if all_tweets_with_smell_triggers['LOC'].iloc[i] == np.nan and all_tweets_with_smell_triggers['loc_pretrained'].iloc[i] != np.nan:
#         all_tweets_with_smell_triggers.LOC.iloc[i] = all_tweets_with_smell_triggers['loc_pretrained'].iloc[i]
# print(len(all_tweets_with_smell_triggers))
#
# all_tweets_with_place = all_tweets_with_smell_triggers[all_tweets_with_smell_triggers['LOC'].notnull()]

#
smell_place = word_embedding_df.dropna()
#
#
smell_place.to_csv("smell_place.csv", index=False)
all_tweets_with_place.to_csv("all_tweets_with_place.csv", index=False)
word_embedding_df.to_csv("word_embedding_df.csv", index=False)



import pandas as pd
import json
import numpy as np

with open('new_word_embeddings_smell_related.json') as f:
  embeddings_dict = json.load(f)

remove_unrelated_smell=['la', 'intro', 'pls', 'fav', 'again', 'off', 'in', 'up', 'back', 'out', 'keep', 'come', 'this', 'hear',
                        'already', 'at', 'need','more', 'want', 'ever', 'never', 'does', 'which', 'both', 'are', 'too', 'much',
                        'cannot', 'rather', 'has', 'these', 'enough', 'has', 'actually', 'do', 'ah', 'cn', 'eel', 'd','e','lau',
                        'hahhaa', 'hei', 'kept', 'gave','bc','sees', 'tt', 'cc', 'bd', 'rt', 'hm','liao', 'skl','ltr','tmr']

id_list = [str(i) for i in list(embeddings_dict)]

# print(id_list[0])
# print(list(embeddings_dict.get(id_list[0]))[0])
print(len(id_list))

for i in range(len(id_list)):
  if list(embeddings_dict.get(id_list[i]))[0] in remove_unrelated_smell:
    embeddings_dict.pop(id_list[i])

id_list_new = [str(i) for i in list(embeddings_dict)]
print(len(id_list_new))

embeddings_dict_reduced = embeddings_dict

with open('embeddings_dict_reduced.json', 'w') as fp:
  json.dump(embeddings_dict_reduced, fp, indent=4)



# tweets_smell = pd.read_csv("tweets_smell")
#
#
#
#






#updating new smell triggers list
# new_triggers = ['pungent', 'airrporrrttt']



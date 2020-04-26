import setup
import json
tweets_smell=setup.tweets_smell
words = setup.words

word_embedding_dicts1 = setup.get_word_embeddings(tweets_smell,0,300)



# word_embedding_dicts={}
# for i in range(300,1000):
#     for key in dicts.keys():
#         if key == tweets_smell['all_triggers'].iloc[i]:
#             id = tweets_smell['id'].iloc[i]
#             word_embedding_dict={id:{key:dicts.get(key)}}
#             word_embedding_dicts.update(word_embedding_dict)

            # id = tweets_smell['id'].iloc[i]
            # word_embedding_dict={id:dic}


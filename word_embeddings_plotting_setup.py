from scipy import spatial
from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import json
import pandas as pd
from sklearn.cluster import KMeans
import add_remove_triggers
from sklearn.datasets import make_blobs

# import nltk
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()


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

def getList(dict):
    return dict.keys()

words = getList(word_embedding_pairs)

vectors = [word_embedding_pairs[word] for word in words]


#Propressing with smell dictionary
# smell_dict = {}
# with open("smell_dictionary_quercia.txt", 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.split()
#         words = ["".join(v.split()) for v in values]
#         token = words[0]
#         code = words[1]
#         smell_dict[token] = code
#
# def getList(dict):
#     return dict.keys()
# words = getList(smell_dict)
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem.porter import PorterStemmer
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# words_lem= [lemmatizer.lemmatize(t) for t in words]
# words_lem_= [lemmatizer.lemmatize(t) for t in smell_dict.keys()]
# smell_category = {}
# for key in smell_dict.keys():
#     if key in words_lem:
#         smell_category[key] = smell_dict.get(key)
#remove duplicates in list
# new_dict = dict.fromkeys(words_lem)
# words_processed = []
# for key in new_dict.keys():
#     words_processed.append(key)

# smell_embedding = {}
# for key in embeddings_dict.keys():
#     if key in words_processed:
#         smell_embedding[key] = embeddings_dict.get(key)


tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
vector_2d = tsne_model.fit_transform(vectors)


# fig, ax = plt.subplots(figsize=(20, 20))
# plt.scatter(vector_2d[:, 0], vector_2d[:, 1], color="red")
# for label, x, y in zip(words, vector_2d[:, 0], vector_2d[:, 1]):
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
# # plt.show()

df =pd.DataFrame(vectors, words)
x = df.iloc[:,0:50].values

#using Elbow method to  find the optimal number of clusters in a dataset (df).
# Error =[]
# for i in range(1, 50):
#     Kmeans = KMeans(n_clusters = i).fit(x)
#     model = Kmeans.fit(x)
#     Error.append(Kmeans.inertia_)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,8))
# plt.plot(range(1, 50), Error)
# plt.title('Elbow method')
# plt.xlabel('No of clusters')
# plt.ylabel('Error')
# # plt.show()

# labels = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20)
#
# colors = ("slategrey", "tan", "red", "yellowgreen", "aqua",
#           "pink", "yellow", "black","wheat","darkslategrey",
#           "thistle", "darkseagreen", "darkred","palevioletred","greenyellow",
#           "orange", "aquamarine", "slateblue", "purple", "olive")
# groups = ("1", '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20')
#



# kmeans = KMeans(n_clusters = 20, random_state=0)
# y_means20 = kmeans.fit_predict(x)
# centroids = kmeans.cluster_centers_
# #transform dimension of centroids from 50 to 2 for plotting purpose
# Y_centroids = tsne_model.fit_transform(centroids[:20])
# fig, ax = plt.subplots(figsize=(30, 30))
# plt.scatter(vector_2d[:, 0], vector_2d[:, 1], c=y_means20, cmap = 'rainbow')
# for label, x, y in zip(words, vector_2d[:, 0], vector_2d[:, 1]):
#     plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords="offset points")


# plt.savefig('word_embeddings_smell_reduced_plot.png')

word_embedding_smell_place_sentiment = pd.read_csv("word_embedding_smell_place_sentiment.csv")
word_embedding_smell_place_sentiment= word_embedding_smell_place_sentiment[word_embedding_smell_place_sentiment['index'].notnull()]

print(word_embedding_smell_place_sentiment.head())
print(len(word_embedding_smell_place_sentiment))



word_lists = word_embedding_smell_place_sentiment.iloc[:, 0]
word_embedding_smell_place_vectors =word_embedding_smell_place_sentiment.iloc[:, 1:50]
word_embedding_smell_place_vectors = word_embedding_smell_place_vectors.to_numpy().tolist()

print(word_embedding_smell_place_vectors[0])



word_embedding_smell_place_vectors_for_df = [i for i in word_embedding_smell_place_vectors]

print(word_embedding_smell_place_vectors_for_df[0][0])

word_embedding_smell_place_vectors_2d = tsne_model.fit_transform(word_embedding_smell_place_vectors)


df_new =pd.DataFrame(word_embedding_smell_place_vectors_2d, word_lists)

print(df_new.head())
print(len(df_new))

#Truning sentiment score to category for class labels
sentiment = []
for i in range(len(word_embedding_smell_place_sentiment['sentiment_scores'])):
    if word_embedding_smell_place_sentiment['sentiment_scores'].iloc[i] > 0:
        sentiment.append('positive')
    elif word_embedding_smell_place_sentiment['sentiment_scores'].iloc[i] < 0:
        sentiment.append('negative')
    else:
        sentiment.append('netural')

print(len(sentiment))


df_new['sentiment'] = sentiment
# print(df_new.head(10))


positive = df_new[df_new['sentiment'] == 'positive']
positive=positive.drop(columns='sentiment')
negative = df_new[df_new['sentiment'] == 'negative']
negative=negative.drop(columns='sentiment')
netural = df_new[df_new['sentiment'] == 'netural']
netural=netural.drop(columns='sentiment')


positive_label = positive.reset_index(level=0)
positive_label = positive_label['index']
negative_label = negative.reset_index(level=0)
negative_label = negative_label['index']
netural_label = netural.reset_index(level=0)
netural_label=netural_label['index']
#
#
data = (positive, negative,netural)
colors =("aquamarine", "slateblue", "purple")
groups = ("positive", "negative", "netural")

labels =(positive_label,negative_label,netural_label)





fig = plt.figure(figsize=(25, 25))
ax = fig.add_subplot(1,1,1)
for d, color, group, label in zip(data, colors, groups, labels):
    x = d[0]
    y = d[1]
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none',label=group)
    for i in range(len(label)):
        ax.annotate(label[i],
                    xy=(x[i], y[i]),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

plt.title('smell_sentiment_plot')

plt.legend(loc=2)
# plt.show()
plt.savefig('smell_sentiment_plot.png')









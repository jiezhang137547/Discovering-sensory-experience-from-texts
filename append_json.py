import json

with open('word_embeddings_smell_related00.json') as f:
    new= json.load(f)

with open('word_embeddings_smell_related.json') as t:
    to_append = json.load(t)

with open('word_embeddings_smell_related1.json') as g:
    to_append1 = json.load(g)

with open('word_embeddings_smell_related2.json') as h:
    to_append2 = json.load(h)

with open('word_embeddings_smell_related4.json') as j:
    to_append4 = json.load(j)

with open('word_embeddings_smell_related5.json') as k:
    to_append5 = json.load(k)


with open('new_word_embeddings_smell_related.json', 'w') as fp:
    new.update(to_append)
    new.update(to_append1)
    new.update(to_append2)
    new.update(to_append4)
    json.dump(new, fp, indent=4)


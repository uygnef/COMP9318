from nltk.corpus import wordnet as wn

for synset in wn.synsets('apples'):
    print(synset, synset.hypernyms())

print(wn.synsets('edible_fruit')[0].lemma())

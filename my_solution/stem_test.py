from nltk.stem.lancaster import LancasterStemmer


nt = LancasterStemmer()
print(nt.stem("APPLE"))
a = "APPLE"
print(a.lower())
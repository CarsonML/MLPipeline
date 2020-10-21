import nltk
from sklearn.feature_extraction.text import CountVectorizer

class POSTaggerModel():
	def __init__(self, pos_ngram, vec_ngram_range):
		self.pos_ngram = pos_ngram
		self.vec_ngram_range = vec_ngram_range
		self.tfidf = CountVectorizer(ngram_range=vec_ngram_range)
	def fit(self, data):
		#get POS
		data = self.makePOS(data)
		#Fit the model to the parts of speech corpus
		self.tfidf.fit(data)
	def transform(self, data):
		#this converts the data to parts of speech then transforms
		data = self.makePOS(data)
		return self.tfidf.transform(data)
	def makePOS(self, data):
		#GEt list w/ parts of speech and words, then only take the parts of speech
		pos_final = []
		for item in data:
			pos = nltk.pos_tag(nltk.word_tokenize(item))
			pos_data = ""
			for item in pos:
				pos_data += (item[1])
				pos_data += " "
			pos_data = pos_data[:-1]
			print(pos_data)
			pos_final.append(pos_data)
		return(pos_final)


# m = POSTaggerModel(1, [1,3])
# corpus = [  'This is the first document.',  'This document is the second document.', 'And this is the third one.',   'Is this the first document?',]
# m.fit(corpus)
# _ = m.transform(corpus)




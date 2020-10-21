import nltk
import numpy as np

class GloveModel():
	def __init__(self):
		self.embeddings_dict = {}
	def fit(self, data):
		#load embeddings
		with open("./glove.6B/glove.6B.50d.txt", 'r') as f:
		    for line in f:
		        values = line.split()
		        word = values[0]
		        vector = np.asarray(values[1:], "float32")
		       	self.embeddings_dict[word] = vector
	def transform(self, data):
		#this converts the data to vectors
		new_data = []
		for item in data:
			initial = np.zeros(1,50)
			for word in nltk.word_tokenize(item):
				new_vector = np.square(np.asarray(self.embeddings_dict[word]))
				initial = np.add(initial, new_vector)
			new_data.append(list(initial))
		return(new_data)


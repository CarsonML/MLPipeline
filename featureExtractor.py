from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from posTaggerModel import POSTaggerModel
import numpy as np
from os import path
import pickle
class FeatureExtractor():
	def __init__(self, features, path, kwargs):
		self.features = features
		self.path = path
		self.kwargs = kwargs
	def buildVectorizer(self, data):
		#Create and train an instance of each of the models specified in self.features
		if path.exists(self.path + "vectorizers.pickle"):
			with open(self.path + "vectorizers.pickle", "rb") as file:
				self.model_list = pickle.load(file)
		else:
			self.model_list = []
			for x in range(len(self.features)):
				current_vectorizer = self.resolve(self.features[x])(**self.kwargs[x])
				current_vectorizer.fit(data)
				self.model_list.append(current_vectorizer)
			with open(self.path + "vectorizers.pickle", "wb+") as file:
				pickle.dump(self.model_list, file)
	def process(self, data):
		#Transform data and store in in a list that will be concatenated later
		transformed_list = []
		for model in self.model_list:
				transformed_data = model.transform(data)
				transformed_list.append(transformed_data)
				print(transformed_data.shape)
				print(transformed_data[1,:])
		return transformed_list
	def resolve(self, step):
		configurations = {'tfidf':TfidfVectorizer, 'pos': POSTaggerModel, 'count':CountVectorizer}
		#These asserts will raise an error if the model string is not found
		assert step in configurations
		return configurations[step]

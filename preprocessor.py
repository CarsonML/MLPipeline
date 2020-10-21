import math
from os import path
import pickle
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
class Preprocessor():
	def __init__(self):
		pass
	def fillnan(self, current_data):
		#Iterate through the data, and replace all NaNs with empty strings
		for x in range(len(current_data)):
			if current_data[x] is None:
				current_data[x] = ''
	def lowercase(self, current_data):
		#Recursively lowercase every item embedded in the data
		x = 0
		for item in current_data:
			print(x/len(current_data))
			if type(item) == list:
				self.lowercase(item)
			if type(item) == str:
				current_data[current_data.index(item)] = item.lower()
			x += 1
	def removeStopwords(self, current_data):
		#Remove stopwords somewhatr recursely
		stop_words = set(stopwords.words('english')) 
		new_data = []
		for item in current_data:
			if type(item) == list:
				self.removeStopwords(item)
			if type(item) == str:
				word_tokens = word_tokenize(item) 
				filtered_sentence = [w for w in word_tokens if not w in stop_words] 
				untokenized = ""
				for item in filtered_sentence:
					untokenized += (item + " ")
				untokenized = untokenized[:-1]
				new_data.append(untokenized)
		return(new_data)
  
	def process(self,step, data):
		#Given preprocessing step, implement that step on the data
		process_func = self.resolve(step)
		process_func(data)
		return data
	def resolve(self, step):
		#Resolve a specific config string to function pointers or list thereof
		configurations = {'fillnan':self.fillnan,'lowercase':self.lowercase}
		#These asserts will raise an error if the config string is not found
		assert step in configurations
		return configurations[step]



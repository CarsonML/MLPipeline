from dataLoader import DataLoader
import json
from sklearn.model_selection import train_test_split
class JSONLoader(DataLoader):
	def __init__ (self):
		pass
	def load(self, path):
		X = []
		y = []
		with open(path) as f:
			for line in f:
				try:
					data = json.loads(line)
					X.append(data['reviewText'])
					y.append(data['overall'])
				except:
					print('Seems like you had one bad JSON line. Not necesarrily a problem, but if it happens for all lines chances are somethings wrong')
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
		return(X_train, y_train, X_test, y_test)
1

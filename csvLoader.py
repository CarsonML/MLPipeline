from dataLoader import DataLoader
import csv
from sklearn.model_selection import train_test_split
class CSVLoader(DataLoader):
	def __init__(self):
		pass
	def load(self, path):
		#Load CSV, Extract X (input) and y (output), then train-test split and return
		with open(path) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			X = []
			y = []
			for row in list(csv_reader)[1:1000]:
				X.append(row[2])
				y.append(row[3])
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
		return(X_train, y_train, X_test, y_test)
		
from dataLoader import DataLoader
class WiliLoader(DataLoader):
	def __init__(self):
		pass
	def load(self, path):
		#Load all 4 files and split them into lists
		#NOTE: the [0:15000] selection on the data is needed for my slow computer and limited ram, it should be removed for a fast computer
		with open(path + "/x_train.txt", "r+") as file:
			train_data = file.read()
		train_data = train_data.split("\n")
		print(len(train_data))
		train_data = train_data[0:500]
		with open(path + "/y_train.txt", "r+") as file:
			train_lables = file.read()
		train_lables = train_lables.split("\n")
		train_lables = train_lables[0:500]
		with open(path + "/x_test.txt", "r+") as file:
			test_data = file.read()
		test_data = test_data.split("\n")
		test_data = test_data[0:500]
		with open(path + "/y_test.txt", "r+") as file:
			test_lables = file.read()
		test_lables = test_lables.split("\n")
		test_lables = test_lables[0:500]
		return train_data, train_lables, test_data, test_lables
		

import os
import json
import datetime
import pickle

import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


from dataLoader import DataLoader
from wiliLoader import WiliLoader
from csvLoader import CSVLoader
from preprocessor import Preprocessor
from featureExtractor import FeatureExtractor
from os import path

class Pipeline(object):

    def __init__(self, configFile):
        #Load the specific configuration file
        self.config = json.load(open(configFile, 'r'))

    def execute(self):
        #Execute the pipeline
        print('Loading Data - ' + self.timestamp())
        train_data, train_labels, test_data, test_labels = self.loadData()
        print('Preprocessing Data - ' + self.timestamp())
        clean_train, clean_test = self.preprocessData(train_data, test_data)
        print('Extracting Features - ' + self.timestamp())
        train_vectors, test_vectors = self.extractFeatures(clean_train, clean_test)
        print('Training Model - ' + self.timestamp())
        model = self.fitModel(train_vectors, train_labels)
        print('Evaluating Model - ' + self.timestamp())
        self.evaluate(model, test_vectors, test_labels)

    def loadData(self):
        #Load the data as specified by the config file
        dataLoader = self.resolve('dataLoader', self.config['dataLoader'])()
        return dataLoader.load(self.config['dataPath'])

    def preprocessData(self, train_data, test_data):
        #Preprocess the data as specified in the config file
        preprocessor = Preprocessor()
        #Check for ALready done work
        if path.exists(self.config['preprocessingPath'] + "Wili1.pickle"):
            with open(self.config['preprocessingPath'] + "Wili1.pickle", "rb") as file:
                train_data, test_data = pickle.load(file)
        else:
            for step in self.config['preprocessing']:
                train_data = preprocessor.process(step, train_data)
                test_data = preprocessor.process(step, test_data)
            with open(self.config['preprocessingPath']  + "Wili1.pickle", "wb+") as file:
                pickle.dump((train_data, test_data), file)
        return train_data, test_data

    def extractFeatures(self, train_data, test_data):
        #Extract Features and pass them as concatenated arrays
        fe = FeatureExtractor(self.config['features'],  self.config['featurePath'], self.config['featureKwargs'])
        fe.buildVectorizer(train_data)
        #Check for ALready done work
        if path.exists(self.config['featurePath']+ "train_data.pickle"):
            print("here's the error?")
            with open(self.config['featurePath'] + "train_data.pickle", "rb") as file:
                train_vectors = pickle.load(file)
        else:
            train_vectors = fe.process(train_data)
            with open(self.config['featurePath']+ "train_data.pickle", "wb+") as file:
                pickle.dump(train_vectors, file)
        if len(train_vectors) > 1:
            print("took option A")
            train_vectors = numpy.concatenate(train_vectors, axis=1)
        else:
            print("took option B")
            train_vectors = train_vectors[0]
        print(train_vectors.shape)
        print(train_vectors[1,:])
        #Check for ALready done work
        if path.exists(self.config['featurePath']+ "test_data.pickle"):
            with open(self.config['featurePath'] + "test_data.pickle", "rb") as file:
                test_vectors = pickle.load(file)
        else:
            test_vectors = fe.process(test_data)
            with open(self.config['featurePath']+ "test_data.pickle", "wb+") as file:
                pickle.dump(test_vectors, file)
        if len(test_vectors) > 1:
            test_vectors = numpy.concatenate(test_vectors, axis=1)
        else:
            test_vectors = test_vectors[0]
        return train_vectors.toarray(), test_vectors.toarray()

    def fitModel(self, train_vectors, train_labels):
        #Fit the model specified in the config file (with specified args)
        #Check for ALready done work
        if path.exists(self.config['modelPath']+ "model.pickle"):
            with open(self.config['modelPath'] + "model.pickle", "rb+") as file:
                model = pickle.load(file)
            return model
        else:
            model = self.resolve('model', self.config['model']) 
            model = model(**self.config["modelKwargs"])
            model.fit(train_vectors, train_labels)
            with open(self.config['modelPath'] + "model.pickle", "wb+") as file:
                pickle.dump(model, file)
            return model
    def evaluate(self, model, test_data, test_labels):
        #Evaluate using the metrics specified in the config file
        print(model)
        #Check for ALready done work
        if path.exists(self.config['metricPath']+ "metrics.pickle"):
            with open(self.config['metricPath'] + "metrics.pickle", "rb") as file:
                results = pickle.load(file)
 
        else:         
            predictions = model.predict(test_data)
            results = {}
            for metric in self.config['metrics']:
                results[metric] = self.resolve('metrics', metric)(predictions, test_labels, **self.config['metricsKwargs'][self.config['metrics'].index(metric)])
            with open(self.config['metricPath'] + "metrics.pickle", "wb+") as file:
                pickle.dump(results, file)
        
        self.output(results)
        print(results)

    def output(self, results):
        output_file = os.path.join(self.config['outputPath'], self.config['experimentName'])
        F = open(output_file, 'w')
        F.write(json.dumps(self.config) + '\n')
        for metric in results:
            F.write(metric + ',%f\n' % results[metric])
        F.close()

    def resolve(self, category, setting):
        #Resolve a specific config string to function pointers or list thereof
        configurations = {'dataLoader': {'baseLoader': DataLoader,
                                         'WiliLoader': WiliLoader,
                                         'CSVLoader': CSVLoader},
                          'model': {'Naive Bayes': GaussianNB,
                                    'SVM': SVC,
                                    'Logistic Regression': LogisticRegression,
                                    'Decision Tree' : DecisionTreeClassifier},
                          'metrics': {'accuracy': accuracy_score,
                                       'f1': f1_score,
                                       'precision':precision_score,
                                       'recall':recall_score}
                         }
        #These asserts will raise an error if the config string is not found
        assert category in configurations
        assert setting in configurations[category]
        return configurations[category][setting]

    def timestamp(self):
        now = datetime.datetime.now()
        return ('%02d:%02d:%02d' % (now.hour, now.minute, now.second))

if __name__ == '__main__':
    p = Pipeline('config.json')
    p.execute()

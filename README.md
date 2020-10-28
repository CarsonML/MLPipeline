# MLPipeline
Machine Learning Pipeline for text classification, can specify via a config file among a myriad of different machine learning models (SVM, Decision Tree, Logistic Regression, Naive Bayes) and select different features/preprocessing steps to use. 

# Overview
This is a machine learning classfier pipeline I built designed for customizability. The goal was to have a pipeline that could take pretty much any input text file with text input data and text of its corresponding class, then train a classifier in the way specified by the config file, evalated on a chosen metric. It also saves whenever it completes a step in case of error. Here's a list of all the things the config file supports (although the beauty of a pipeline like this, and kind of the whole point, is that new options and models and features can be added easily with few lines of code):

### File Types:
CSV, JSON, and Wili (Custom for one of the databases, bascially just data split into train and test files and separated by newline

### Preprocessing:
Fill Nan Values (fillnan), Lowercase, remove stopwords

### Vectorizing/Features:
Count Vectorizer, TFIDF Vectorizer, Parts of Speech Tags, GloVe (BROKEN, for now)

### Models
Naive Bayes, SVM, Logistc Regression, Decision Tree

### Metrics
Accuracy, f1, precision, recall

# Config File
The custom model is based off the config file (sample one included) that has the following format:

`{"dataLoader": "CSVLoader",
"dataPath": "./data/test_file", 
"preprocessing": ["fillnan", "lowercase"],
"preprocessingPath": "./preprocessed/", 
"features": ["tfidf"],
"featureKwargs": [{"ngram_range": [1,1], "analyzer": "word"}],
"featurePath": "./features/", "model": "Logistic Regression",
"model": "Logistic Regression",
"modelKwargs": {"penalty":"l2"}, "modelPath": "./models/",
"metrics": ["accuracy"], 
"metricsKwargs": [{}],
"metricPath": "./metrics/",
"outputPath": "./results/",
"experimentName": "csv8.txt"
}`

Most of these are pretty self explanatory, but in the interests of clarity: `dataLoader` is the data format, `CSVLoader` for CSV, `JSONLoader` for JSON, and `WiliLoader` for Wili. `dataPath` is the file path to the input data. `preprocessing` is a list of all the preprocessing steps to run. `preprocessingPath` is the file path where it will store/load the preprocessed results. `features` is the list of features/vectorizers to use (vectors are concatenated for more than one feature). `featureKwargs` is a list of the kwargs correspoinding to each feature. `featurePath` is where it stores the features or looks for them if they have aready been completed. `model` of course tells you which model to use (can only use one, for now), and `modelKwargs` are the kwargs for the model. `metrics` are which metrics to use, and `metricKwargs` is a list of their kwargs. `metricPath` is the filepath to store/load the metrics, and `outputPath` is the file path to store the results. `experimentName` is just text put in the results file.

# Preprocessing 
There are 3 preprocessing options:

**Fill NaN**: ('fillnan' in config): Pretty obvious, fills empty spaces in the data so the pipeline doesn't break. Reccomended to always use.

**Lowercase** ('lowercase' in config): Again fairly obvious, lowercases all the text. Also reccomended to use unless differntation between captial and non captial letters is important

**Remove Stopwords**: ('stopwords' in config): Removes all stopwords from the text (at least what nltk consideres to be stopwords). This one is optional to use.

All preprocessing is done somewhat recusrsively, so there are no issues if the data is lists wihtin lists within lists, it should still work. 

**NOTE!!** the current loader files only take in a certain amount of data and discard the rest. That is because my computer is from 2011 and cannot run anything more. If you're computer can handle it, adjust the list indexes in the loader file to add more data. Or ideally, just take them away if you have enough computing power

# Vectorizing/Features
These specify the way the text is turned into vectors. If multiple ways are selected, then the different vectors for each feature are concatenated together (that's the beatuy of vectors. You can just combine them and they're still a vector). Here are the types supported:

**Count Vectorizer** ('count' in config): the most basic: just counts the number of times each word appears in the text and uses that as the vector

**TFIDF** ('tfidf' in config): A bit fancier than count, counts the number of times each word appears but then normalizes that vector

**Parts of Speech** ('pos' in config): Converts the text into its parts of speech, and then uses a count vectorizer on that. Cool in theory, but this one has never helped me much

**GloVe** (BROKEN, not in config yet): This is all coded but I haven't tested it, and therefore have not added it to the lookup (if someone wanted to try and enable it, just add a reference to the features.py file.Also download the 6B glove file from that link and move that folder to the pipeline folder) But it's probably broken. This is a work in progress, update likely coming soon). Vectorizes each word in the given text using [Stanford's GloVe vectors](https://nlp.stanford.edu/projects/glove/) , then squares, adds, and normalizes them to get the final vector

Count and TFIDF are straight scikitlearn, so look there for kwargs. Most important is probably which n grams to use

# Models
These are all standard machine learning models; I could type up descriptions but Scikitlearn will have a better explanation than I can come up with likely, probably using visuals I can't put in markdown. So if one is interested in how these models work, all one must do is google them

**Naive Bayes**: 'Naive Bayes' in config.

**Support Vector Machine**: 'SVM' in config. 

**Logistic Regression**: 'Logistic Regression' in config.

**Decision Tree**: 'Decision Tree' in config.

All these models are based off of scikitlearn, so look there for kwargs. Currenty, using multiple models and comparing them is not supported. 

I've had the best luck with SVMs and Naive Bayes. And pretty terrible luck with decision trees. But they will all work!

# Metrics
These are again fairly standard ML metrics. Multiple can be used at one time. Again, like the models, they are based on scikitlearn so look there for their kwargs

**Accuracy**: 'accuracy' in config

**F1**: 'f1' in config

**Precision**: 'precision' in config

**Recall**: 'recall' in config

The summary of the model and its metrics is printed to conosle and also saved in the output file

# Use Cases
Really nothing specific. Anything where you have input text and want to classify it into different categories. Some of the things I've used it for are classifying wikipedia articles into their lagnuage (works quite well) or trying to score SAT essays (works quite poorly, probably a good thing). Those two datasets are included. As well as a bonus one about reviews I haven't worked with yet. But the sky's your oyster!





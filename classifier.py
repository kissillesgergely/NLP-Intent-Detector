import gensim
import sklearn
import nltk

from copy import deepcopy
import numpy as np
from sklearn import tree

class Sentences(object):    
	def __init__(self, list):        
		self.list = list
 
	def __iter__(self):
		for doc in self.list:	
			yield nltk.tokenize.word_tokenize(doc)


class Classifier:
	def __init__(self):
		self.prediction_model = None
		self.word2vec_model = None
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None

		self.hyperparameters = {
			'decision_tree_depth': 4,
			'word2vec_size': 200,
			'word2vec_window': 5,
			'word2vec_min_count': 10,
			'word2vec_iterations': 10,
		}

	def train_classifier(self, docs, ground_truths):
		vectorized_docs = []

		for doc in docs:
			document_embeddings = np.zeros(self.word2vec_model.vector_size)
			for word in doc.split():
				if word in self.word2vec_model.wv.vocab:
					document_embeddings += self.word2vec_model.wv[word]
			vectorized_docs.append(document_embeddings)

		fold_num = 5
		fold = int(len(vectorized_docs)/fold_num)
		vectorized_docs, ground_truths = sklearn.utils.shuffle(np.array(vectorized_docs), np.array(ground_truths))
		vectorized_docs = list(vectorized_docs)
		ground_truths = list(ground_truths)    
		accuracies = []
		precisions = []
		recalls = []
		f1s = []

		for i in range(fold_num):
			prediction_model = None
			prediction_model = tree.DecisionTreeClassifier(
				max_depth=self.hyperparameters['decision_tree_depth']
			)

			vectors_copy = deepcopy(vectorized_docs)
			truths_copy = deepcopy(ground_truths)
 
			fold_test_input = vectors_copy[i*fold:(i+1)*fold]
			fold_test_truth = truths_copy[i*fold:(i+1)*fold]

			del vectors_copy[i*fold:(i+1)*fold]
			del truths_copy[i*fold:(i+1)*fold]

			fold_train_input = vectors_copy
			fold_train_truth = truths_copy

			prediction_model.fit(fold_train_input, fold_train_truth)
			predictions = prediction_model.predict(fold_test_input)

			accuracies.append(sklearn.metrics.accuracy_score(predictions, fold_test_truth))
			precisions.append(sklearn.metrics.precision_score(predictions, fold_test_truth, average='weighted'))
			f1s.append(sklearn.metrics.f1_score(predictions, fold_test_truth, average='weighted'))
			recalls.append(sklearn.metrics.recall_score(predictions, fold_test_truth, average='weighted'))

		self.accuracy = 0
		self.precision = 0
		self.f1 = 0
		self.recall = 0
		# Calculating an average from the folds
		for i in range(len(accuracies)):
			self.accuracy  += accuracies[i]/fold_num
			self.precision += precisions[i]/fold_num
			self.f1 += f1s[i]/fold_num
			self.recall += recalls[i]/fold_num

		self.prediction_model = sklearn.tree.DecisionTreeClassifier(
			max_depth=self.hyperparameters['decision_tree_depth']
		)
		self.prediction_model.fit(vectorized_docs, ground_truths)

	def create_word2vec_model(self, sentences):
		data_iterator = Sentences(sentences)
		self.word2vec_model = gensim.models.Word2Vec(
			data_iterator,
			size=self.hyperparameters['word2vec_size'],
			window=self.hyperparameters['word2vec_window'],
			min_count=self.hyperparameters['word2vec_min_count'],
			iter=self.hyperparameters['word2vec_iterations']
		)

	def predict(self, text):
		document_embeddings = np.zeros(self.word2vec_model.vector_size)
		for word in text.split():
				if word in self.word2vec_model.wv.vocab:
					document_embeddings += self.word2vec_model.wv[word]
		
		return self.prediction_model.predict([document_embeddings])


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.utils

import glob
import os.path
import pprint


def load_data():
	'''
	reads all question files and returns - 
	X, y --> q_text, label
	'''

	complexities = dict()
	complexity_file = open("questions-complexity.csv")
	complexity_file.readline()

	for line in complexity_file:
		line = line.strip().split(",")
		complexities[line[0]] = line[-2]
	complexity_file.close()


	question_files = glob.glob("./questions/*.txt")	
	questions = []
	labels = []

	for f in question_files:
		handle = open(f)
		text = handle.read()
		handle.close()
		text = text.split("\n\n")[2]	### + [3] and [4] for Input and Output requirement text

		### replace these with robust nltk fn calls if possible
		text = text.replace("."," ")
		text = text.replace(","," ")
		text = text.replace("!"," ")
		text = text.lower()
		#######################################################
		labels.append(complexities[os.path.basename(f).strip(".txt")])
		questions.append(text)
		
	return questions, labels
def create_vocab(questions):
	'''
	takes list of questions (returned from load_data) and returns a list of words sorted in desending order of frequency
	'''
	word_freq = dict()
	for ques in questions:
		for word in ques.split():
			if word not in word_freq:
				word_freq[word] = 0
			word_freq[word] += 1
	top_words = sorted(word_freq, key = lambda w : word_freq[w], reverse = True)
	global vocab_size
	vocab_size = len(top_words)
	return top_words

def manipulate_data(questions, labels):
	vocab = create_vocab(questions)
	questions_vector = []
	for ques in questions:
		single_question_vector = []
		for word in ques.split():
			single_question_vector.append(vocab.index(word))
		questions_vector.append(single_question_vector)
	
	labels = [0 if l == "Easy" else l for l in labels]
	labels = [1 if l == "Medium" else l for l in labels]
	labels = [2 if l == "Hard" else l for l in labels]
	
	labels_vector = keras.utils.to_categorical(labels)
	
	return questions_vector, labels_vector
	
def build_model(X,y):
	global max_question_length, embedding_vector_length, n_epochs
	X = sequence.pad_sequences(X, maxlen=max_question_length)

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_vector_length))

	model.add(LSTM(10))
	
	model.add(Dense(3, activation = "softmax"))

	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	model.fit(X, y, epochs=n_epochs, batch_size=64, verbose=1)

	return model

def test_model(model, X, y):
	global max_question_length
	X = sequence.pad_sequences(X, maxlen=max_question_length)
	scores = model.evaluate(X, y)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def predict(X):
	pass


if __name__ == "__main__":
	max_question_length = 300
	vocab_size = 0
	embedding_vector_length = 32
	n_epochs = 5
	
	X, y = load_data()
	X, y = manipulate_data(X, y)
	
	model = build_model(X[:1700], y[:1700])
	test_model(model, X[1700:], y[1700:])
	
	
	
	
	
	
	
	
	
	
	
	
	

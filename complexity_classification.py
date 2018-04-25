import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import keras.utils

import nltk
import glob
import os.path
import pprint


def tokenize_and_lower(string):
	return " ".join(nltk.word_tokenize(string)).lower()
	
def load_data():
	'''
	reads all question files and returns - 
	q_text, input, output, label
	'''

	complexities = dict()
	complexity_file = open("questions-complexity.csv", encoding='utf-8')
	complexity_file.readline()

	for line in complexity_file:
		line = line.strip().split(",")
		complexities[line[0]] = line[-2]
	complexity_file.close()


	question_files = sorted(glob.glob("./questions/*.txt"))
	questions = []
	inputs = []
	outputs = []
	labels = []

	for f in question_files:
		handle = open(f, encoding='utf-8')
		text = handle.read()
		handle.close()
		text_split = text.split("\n\n")
		question = text_split[2]	### + [3] and [4] for Input and Output requirement text
		input = text_split[3]
		output = text_split[4]
		
		# Removes 'Input' and 'Output' prefixes
		input = input[len('Input'):]
		output = output[len('Output'):]
		
		question = tokenize_and_lower(question)
		input = tokenize_and_lower(input)
		output = tokenize_and_lower(output)
		
		questions.append(question)
		inputs.append(input)
		outputs.append(output)
		labels.append(complexities[os.path.basename(f).strip(".txt")])
		
	return questions, inputs, outputs, labels
	
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
	vocab_size = len(top_words)
	
	return top_words, vocab_size

def vectorize_data(questions, labels, vocab):
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
	
def build_model(X, y, vocab_size, max_question_length, embedding_vector_length, n_epochs):
	X = sequence.pad_sequences(X, maxlen=max_question_length)

	model = Sequential()
	model.add(Embedding(vocab_size, embedding_vector_length))

	model.add(LSTM(10))
	
	model.add(Dense(3, activation = "softmax"))

	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

	model.fit(X, y, epochs=n_epochs, batch_size=64, verbose=1)

	print('SUMMARY:', model.summary())
	return model

def test_model(model, X, y, max_question_length):
	X = sequence.pad_sequences(X, maxlen=max_question_length)
	scores = model.evaluate(X, y)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def predict(X):
	pass


if __name__ == "__main__":
	max_question_length = 300
	embedding_vector_length = 32
	n_epochs = 5
	
	questions, inputs, outputs, labels = load_data()
	vocab, vocab_size = create_vocab(questions)
	X, y = vectorize_data(questions, labels, vocab)
	
	X_train, y_train = X[:1700], y[:1700]
	X_test, y_test = X[1700:], y[1700:]
	
	print('length:', len(X))
	model = build_model(X_train, y_train, vocab_size, max_question_length, embedding_vector_length, n_epochs)
	test_model(model, X_test, y_test)
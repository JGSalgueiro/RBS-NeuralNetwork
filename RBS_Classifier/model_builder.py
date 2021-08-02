#******************************************************************************
#* RBS_Classifier                                                             *
#* tensorflow (version 2.0 or higher) and kerastuner                          *
#* Author : João Geraldes Salgueiro <joao.g.salgueiro@tecnico.ulisboa.com>    *
#* Developed with the help of C2TN                                            *
#******************************************************************************

import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import os

DATADIR = "/home/jgsalgueiro/Desktop/RBS_ANN/RBS_NeuralNetwork/Training/TrainingSet" #FIX DIRS
TESTDIR = "/home/jgsalgueiro/Desktop/RBS_ANN/RBS_NeuralNetwork/Training/TestSet"     #FIX DIRS
CATEGORIES = ["ag5au", "ag10au", "ag15au", "ag20au", "ag25au", "ag30au", "ag35au", "ag40au", "ag45au", "ag50au", "ag55au", "ag60au", "ag65au", "ag70au", "ag75au", "ag80au", "ag85au", "ag90au", "ag95au", "ag100au"]
CLASSIFICATION = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0,90,0.95,1.00]
PEAK = 0.5
CHANNELS = 944
MAX_DATA = 0
MAX_FIT = 0

MIN_DATA = 9999999
MIN_FIT = 999999


def sigmoide(x):
	return 1/(1 + np.exp(-x))


def extract_number(input):
	return float(input)

def extract_number_v2(input):
	x = input.split("E")
	if(x[1][0] == 'x'): 
		return float(x[0]) * (10 ** float(x[1]))
	else:
		return float(x[0]) * (10 ** -float(x[1]))


def load_sets():
	global train_spectra
	global train_label
	global test_spectra
	global test_label
	global MAX_DATA, MAX_FIT, MIN_FIT, MIN_DATA
	labels = []
	spectras = []
	labels_test = []
	spectras_test = []
	#for each dir load training sets
	for categorie in CATEGORIES:
		path = os.path.join(DATADIR, categorie)
		for spectra in os.listdir(path):
			spetra_data = []
			f = open(os.path.join(path,spectra))
			for line in (f.readlines() [-CHANNELS:]):
				x = line.split()
				channel = x[0]
				data = x[1]
				fit = x[2]

				fit_num = sigmoide(extract_number(fit))
				spetra_data.append(fit_num)

				if fit_num > MAX_FIT:
					MAX_FIT = fit_num
				elif fit_num < MIN_FIT:
					MIN_FIT = fit_num

			#data.append(spetra_data)
			spectras.append(spetra_data)
			#labels.append(CLASSIFICATION[CATEGORIES.index(categorie)] * 100)
			labels.append(CATEGORIES.index(categorie))

	for categorie in CATEGORIES:
		path = os.path.join(TESTDIR, categorie)
		for spectra in os.listdir(path):
			spetra_data = []
			f = open(os.path.join(path,spectra))
			for line in (f.readlines() [-CHANNELS:]):
				x = line.split()
				channel = x[0]
				data = x[1]
				fit = x[2]

				fit_num = sigmoide(extract_number_v2(fit))
				spetra_data.append(fit_num)

				if fit_num > MAX_FIT:
					MAX_FIT = fit_num
				elif fit_num < MIN_FIT:
					MIN_FIT = fit_num

			#data.append(spetra_data)
			spectras_test.append(spetra_data)
			#labels_test.append(CLASSIFICATION[CATEGORIES.index(categorie)] * 100)
			labels_test.append(CATEGORIES.index(categorie))


	print("Loaded training inputs from : " + categorie + ";")

	train_label = np.array(labels)
	train_spectra = np.array(spectras)
	test_label = np.array(labels_test)
	test_spectra = np.array(spectras_test)
	print("Finished loading sets;")


def normalize(num):
	global train_spectra
	return num


def poisson_noise():
	s = np.random.poisson(2, 1024)
	print(s)
	return 0

def build_model_v1():
	global train_spectra
	global train_label
	global test_spectra
	global test_label


	model = keras.Sequential([
	keras.layers.Flatten(),
    keras.layers.Dense(1024, input_dim = 1024, activation = 'relu'),
    keras.layers.Dense(512, activation = 'relu'),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(20, activation = 'softmax')
    ])

	"""
	model = keras.Sequential([
	keras.layers.Dense(1024, input_dim = 1024, activation = 'linear'),
	keras.layers.Dense(1024, activation='tanh'),
	keras.layers.Dense(512, activation='tanh'),
	keras.layers.Dense(256, activation='tanh'),
	keras.layers.Dense(1, activation='sigmoid')
	])

	sgd = keras.optimizers.RMSprop(lr=0.001)
	model.compile(loss=keras.losses.mean_squared_error,optimizer=sgd, metrics=['mae'])"""

	model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
	#model.compile(optimizer='adam',loss='poisson', metrics=['accuracy'])

	model.fit(train_spectra, train_label, epochs = 50, batch_size = 128 , verbose=1, validation_data = (test_spectra, test_label))
	return model



def main(): 
	global train_spectra
	global train_label
	global test_spectra
	global test_label
	load_sets()
	
	model = build_model_v1()
	model.save('./best_model')

	
	#loaded_model = keras.models.load_model('./best_model')
	print(train_spectra[1])
	print(train_spectra.shape)

	"""
	for i in range(10):
		load_spectra = np.expand_dims(test_spectra[i], axis=0)
		results = model.predict(load_spectra)

		print(test_label[i])
		#print(results)
		
		print("EXPECTED : " , CATEGORIES[test_label[i]])
		print("OBTAINED : " , CATEGORIES[np.argmax(results)])
	"""

	model.evaluate(test_spectra, test_label)
	


if __name__ == "__main__":
    main()
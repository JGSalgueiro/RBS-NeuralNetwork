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
CLASSIFICATION = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
PEAK = 0.5
CHANNELS = 1024
MAX_DATA = 0
MAX_FIT = 0

MIN_DATA = 9999999
MIN_FIT = 999999


def sigmoide(x):
	return 1/(1 + np.exp(-x))


def extract_number(input):
	return float(input)


def load_sets():
	global train_spectra
	global train_label
	global MAX_DATA, MAX_FIT, MIN_FIT, MIN_DATA
	labels = []
	spectras = []
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
			#labels.append(CLASSIFICATION[CATEGORIES.index(categorie)] / 100)
			labels.append(CATEGORIES.index(categorie))


	print("Loaded training inputs from : " + categorie + ";")
	train_label = np.array(labels)
	train_spectra = np.array(spectras)
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
	model = keras.Sequential([
	keras.layers.Flatten(),
	keras.layers.Dense(1024, input_dim = 1024, activation = 'relu'),
	keras.layers.Dense(512, activation = 'relu'),
	keras.layers.Dense(256, activation = 'relu'),
	keras.layers.Dense(20, activation = 'softmax')
	])

	model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
	#model.compile(optimizer='adam',loss='poisson', metrics=['accuracy'])

	model.fit(train_spectra, train_label, epochs = 5, batch_size = 20)
	return model

def build_model(hp):
	model = keras.Sequential()
	model.add(keras.layers.Flatten()),
	model.add(keras.layers.Dense(512, input_shape = (2,1024,1), activation = 'relu'))
	model.add(keras.layers.Dense(256, activation = 'relu'),)

	

	"""for i in range(hp.Int("Conv layers", min_value = 1, max_value = 4)):
		model.add(keras.layers.Conv1D(hp.Choice(f"layer_{i}_filters", [16,32,64]) ,3 ,activation = 'relu'))

	model.add(keras.layers.MaxPool2D(2,2))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Flatten())

	model.add(keras.layers.Dense(hp.Choice("Dense_layer", [64, 128, 256, 512, 1024]), activation = 'relu'))
	model.add(keras.layers.Dense(3, activation = 'softmax'))"""

	model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
	return model


def create_model():
	global train_spectra
	global train_label
	#train_spectra = pad_sequences(train_spectra, maxlen = 20)
	print(train_label)
	tuner = RandomSearch(build_model, objective = 'val_accuracy', max_trials = 5)
	tuner.search(train_spectra, train_label, validation_data = (train_spectra, train_label), epochs = 10, batch_size = 32)
	return tuner.get_best_models()[0]


def main(): 
	global train_spectra
	global train_label
	i = 1
	load_sets()
	
	model = build_model_v1()
	model.save('./best_model')

	
	loaded_model = keras.models.load_model('./best_model')
	print(train_spectra[1])
	print(train_spectra.shape)

	
	for i in range(33):
		load_spectra = np.expand_dims(train_spectra[i], axis=0)
		results = loaded_model.predict(load_spectra)
		
		print("EXPECTED : " , CLASSIFICATION[train_label[i]] ,"%")
		print("OBTAINED : " , CLASSIFICATION[np.argmax(results)], "%")
	
	


if __name__ == "__main__":
    main()
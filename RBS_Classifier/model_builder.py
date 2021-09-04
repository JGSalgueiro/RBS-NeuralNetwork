#******************************************************************************
#* RBS_Model Builder                                                          *
#* tensorflow (version 2.0 or higher) and kerastuner                          *
#* Author : João Geraldes Salgueiro <joao.g.salgueiro@tecnico.ulisboa.com>    *
#* Developed with the help of C2TN                                            *
#******************************************************************************

import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf 
import random
import pickle
from datetime import datetime
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import os

DATADIR = "Training/TrainingSet" 
TESTDIR = "Training/TestSet"     
CATEGORIES = ["cu5au", "cu10au", "cu15au", "cu20au", "cu25au", "cu30au", "cu35au", "cu40au", "cu45au", "cu50au", "cu55au", "cu60au", "cu65au", "cu70au", "cu75au", "cu80au", "cu85au", "cu90au", "cu95au", "cu100au"]
CLASSIFICATION = [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0,90,0.95,1.00]

CHANNELS = 1024


def sigmoide(x):
	return 1/(1 + np.exp(-x))


def find_MAX(spectra_set):
	MAX = 0

	for spectrum in spectra_set:
		if np.amax(spectrum) > MAX:
			MAX = np.amax(spectrum)

	
	return MAX


def normalize(spectra_set):
	sets = []

	for i in spectra_set:
		sets.append(np.true_divide(i, np.amax(i)))

	return np.array(sets).reshape(-1, 1024, 1)


def extract_number(input):
	return float(input)

def extract_number_v2(input):
	x = input.split("E")
	if(x[1][0] == '+'): 
		return float(x[0]) * (10 ** float(x[1][1:]))
	elif(x[1][0] == '-'):
		return float(x[0]) * (10 ** -float(x[1][1:]))


def load_sets():
	global train_spectras
	global train_labels

	training_dataset = []
	test_dataset = []

	#for each dir load training sets
	for categorie in CATEGORIES:
		path = os.path.join(DATADIR, categorie)
		for spectra in os.listdir(path):
			spetra_data = []
			f = open(os.path.join(path,spectra))
			for line in (f.readlines()[-CHANNELS:]):
				x = line.split()
				channel = x[0]
				data = x[1]
				fit = x[2]

				fit_num = extract_number(fit)
				spetra_data.append(fit_num)

			training_dataset.append([np.array(spetra_data), CATEGORIES.index(categorie)])


	#for each dir load testing sets
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

				fit_num = extract_number_v2(fit)
				spetra_data.append(fit_num)

			test_dataset.append([np.array(spetra_data), CATEGORIES.index(categorie)])


	random.shuffle(training_dataset)
	train_spectras = []
	train_labels = []
	test_spectras = []
	test_labels = []

	for spectra, label in training_dataset:
		train_spectras.append(spectra)
		train_labels.append(label)

	for spectra, label in test_dataset:
		test_spectras.append(spectra)
		test_labels.append(label)

	train_spectras = np.array(train_spectras).reshape(-1, 1024, 1)
	test_spectras = np.array(test_spectras).reshape(-1, 1024, 1)

	pickle_out = open("Train_Spectra.pickle", "wb")
	pickle.dump(train_spectras, pickle_out)
	pickle_out.close()

	pickle_out = open("Train_Label.pickle", "wb")
	pickle.dump(train_labels, pickle_out)
	pickle_out.close()

	pickle_out = open("Test_Spectra.pickle", "wb")
	pickle.dump(test_spectras, pickle_out)
	pickle_out.close()

	pickle_out = open("Test_Label.pickle", "wb")
	pickle.dump(test_labels, pickle_out)
	pickle_out.close()



def poisson_noise():
	s = np.random.poisson(2, 1024)
	print(s)
	return 0

def build_model():
	global train_spectra
	global train_label
	global test_spectra
	global test_label

	model = keras.Sequential([
	keras.layers.Conv1D(512, 3,activation = 'relu', input_shape = (1024,1)),
	keras.layers.MaxPooling1D(2),
	keras.layers.Conv1D(256, 3,activation = 'relu', input_shape = (1024,1)),
	keras.layers.MaxPooling1D(2),
	keras.layers.Flatten(),
    keras.layers.Dense(64),
    keras.layers.Dense(40, activation = 'softmax'),
	])

	model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

	model.fit(train_spectra, train_label, epochs = 10, batch_size = 128, validation_split = 0.05)
	return model



def main(): 
	global train_spectra
	global train_label
	global test_spectra
	global test_label

	print(datetime.now().strftime("%H:%M:%S") , " - " , "Loading data ...")
	load_sets()
	
	pickle_in = open("Train_Spectra.pickle", "rb")
	train_spectra = np.array(pickle.load(pickle_in))
	pickle_in = open("Train_Label.pickle", "rb")
	train_label = np.array(pickle.load(pickle_in))
	pickle_in = open("Test_Spectra.pickle", "rb")
	test_spectra = np.array(pickle.load(pickle_in))
	pickle_in = open("Test_Label.pickle", "rb")
	test_label = np.array(pickle.load(pickle_in))

	train_spectra = normalize(train_spectra)
	test_spectras = normalize(test_spectra)


	print(datetime.now().strftime("%H:%M:%S") , " - " , "Loaded training spectras with shape : " ,train_spectra.shape)
	print(datetime.now().strftime("%H:%M:%S") , " - " , "Loaded testing spectras with shape  : " ,test_spectra.shape)
	print(datetime.now().strftime("%H:%M:%S") , " - " , "Will now start training model. ")
	model = build_model()
	model.save('./best_model')
	print(datetime.now().strftime("%H:%M:%S") , " - " , "Saved model : SUCESS")
	print(datetime.now().strftime("%H:%M:%S") , " - " , "Model Training - SUCESS : will now avaluate the performance. ")
	


	#loaded_model = keras.models.load_model('./best_model')
	print(model.evaluate(train_spectra, train_label))




if __name__ == "__main__":
    main()

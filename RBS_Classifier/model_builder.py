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
from tensorflow import keras
import os

DATADIR = "/home/jgsalgueiro/Desktop/RBS_ANN/RBS_NeuralNetwork/Training"
CATEGORIES = ["ag5au", "ag10au", "ag15au", "ag20au", "ag25au", "ag30au", "ag35au", "ag40au", "ag45au", "ag50au", "ag55au", "ag60au", "ag65au", "ag70au", "ag75au", "ag80au", "ag85au", "ag90au", "ag95au", "ag100au"]
PEAK = 0.5
CHANNELS = 1024
train_spectra = None
train_label = None

def extract_number(input):
	x = input.split("E")
	if(x[1][0] == 'x'): 
		return float(x[0]) * (10 ** float(x[1]))
	else:
		return float(x[0]) * (10 ** -float(x[1]))


def load_sets():
	labels = []
	spectras = []
	#for each dir load training sets
	for categorie in CATEGORIES:
		path = os.path.join(DATADIR, categorie)
		for spectra in os.listdir(path):
			spetra_data = [[],[]]
			f = open(os.path.join(path,spectra))
			for line in (f.readlines() [-CHANNELS:]):
				x = line.split()
				channel = x[0]
				data = x[1]
				fit = x[2]

				spetra_data[0].append(extract_number(data))
				spetra_data[1].append(extract_number(fit))

			#data.append(spetra_data)
			spectras.append(spetra_data)
			labels.append(categorie)


	print("Loaded training inputs from : " + categorie + ";")
	train_label = np.array(labels)
	train_spectra = np.array(spectras)
	print("Finished loading sets;")
	print(train_spectra.shape)


def normalize(num):
	#Trying to implement Z-Score normalization
	return num


def poisson_noise(data):
	#Add poisson noise to training set 
	return 0


def build_model(hp):
	model = keras.Sequential()
	model.add(keras.layers.AveragePooling2D(6,3, input_shape = (2,1024)))

	for i in range(hp.Int("Conv layers", min_value = 1, max_value = 4)):
		model.add(keras.layers.Conv2D(hp.Choice(f"layer_{i}_filters", [16,32,64]) ,3 ,activation = 'relu'))

	model.add(keras.layers.MaxPool2D(2,2))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Flatten())

	model.add(keras.layers.Dense(hp.Choice("Dense_layer", [64, 128, 256, 512]), activation = 'relu'))
	model.add(keras.layers.Dense(3, activation = 'softmax'))

	model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])

	return model


def create_model():
	#Normalize Data
	#TODO


	#Add poisson Noise
	#TODO


	tuner = RandomSearch(build_model, objective = 'val_accuracy', max_trials = 5)
	tuner.search(train_img, train_label, validation_data = (test_img, test_label), epochs = 10, batch_size = 32)
	return tuner.get_best_models()[0]


def main():
	model = create_model()
	load_sets()


if __name__ == "__main__":
    main()
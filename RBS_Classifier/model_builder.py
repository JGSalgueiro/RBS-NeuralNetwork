******************************************************************************
* RBS_Classifier                                                             *
* tensorflow (version 2.0 or higher) and kerastuner                          *
* Author : João Geraldes Salgueiro <joao.g.salgueiro@tecnico.ulisboa.com>    *
* Developed with the help of C2TN                                            *
******************************************************************************

import numpy as np 
import tensorflow as tf 
from kerastuner.tuners import RandomSearch
from tensorflow import keras

PEAK = 0.5

def poisson_noise(data):
	noise = np.random.poisson(image / 255.0 * PEAK) / PEAK * 255  #create noisy data_set
	return noise

def create_trainning_set():


def build_model(hp):
	#TODO
	return 0

def create_model():
	#TODO
	return 0

def main():
	model = create_model()


if __name__ == "__main__":
    main()



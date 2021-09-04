#******************************************************************************
#* RBS_Classifier                                                             *
#* tensorflow (version 2.0 or higher) and kerastuner                          *
#* Author : João Geraldes Salgueiro <joao.g.salgueiro@tecnico.ulisboa.com>    *
#* Developed with the help of C2TN                                            *
#******************************************************************************
from tensorflow import keras
import numpy as np 
import os

PATH = "To_Classify/"
FILE_NAME = "file.txt"
MODEL_NAME = './best_model'
DATADIR = "Training/TrainingSet" 
TESTDIR = "Training/TestSet"    
CATEGORIES = ["cu5au", "cu10au", "cu15au", "cu20au", "cu25au", "cu30au", "cu35au", "cu40au", "cu45au", "cu50au", "cu55au", "cu60au", "cu65au", "cu70au", "cu75au", "cu80au", "cu85au", "cu90au", "cu95au", "cu100au"]


def load_sets():
	global unclassified_spectra
	global files
	spectra_set = []
	files = []

	#for each dir load training sets

	for spectra in os.listdir(PATH):
		spetrum_data = []
		files.append(spectra)
		f = open(os.path.join(PATH,spectra))

		for line in (f.readlines()[-1024:]):
			x = line.split()
			channel = x[0]
			fit = x[1]

			fit_num = float(fit)
			spetrum_data.append(fit_num)


		normalized_spectrum = [x / max(spetrum_data) for x in spetrum_data]
		spectra_set.append(np.array(normalized_spectrum))

	print(spectra_set)
	unclassified_spectra = np.array(spectra_set).reshape(-1, 1024, 1)


def main(): 
	global unclassified_spectra
	global files

	load_sets()

	file  = open("results", "w+")

	loaded_model = keras.models.load_model(MODEL_NAME)

	results = loaded_model.predict(unclassified_spectra)


	for i in range(len(results)):



		

		file.write(files[i] + " : " + CATEGORIES[np.argmax(results[i])] + "\n")
		print(CATEGORIES[np.argmax(results[i])])

	
	file.close()


if __name__ == "__main__":
    main()

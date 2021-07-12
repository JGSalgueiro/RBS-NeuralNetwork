#******************************************************************************
#* Creates Training Set from given spectra by adding noise                    *
#* tensorflow (version 2.0 or higher) and kerastuner                          *
#* Author : João Geraldes Salgueiro <joao.g.salgueiro@tecnico.ulisboa.com>    *
#* Developed with the help of C2TN                                            *
#******************************************************************************
#******************************************************************************
#* Experimental Conditions for the RBS data                                   *
#* Proton beam with 2MeV of energy                                            *
#* Particle detector angle: 140º                                              *
#* Angle of incidence: 0º                                                     *
#* Detector resolution: 30 keV                                                *
#* Collected charge: 0.1 microCoulomb                                         *                                                                
#******************************************************************************
import os
import numpy as np 

DATADIR = "/home/jgsalgueiro/Desktop/RBS_ANN/RBS_NeuralNetwork/Training/TrainingSet"
TESTDIR = "/home/jgsalgueiro/Desktop/RBS_ANN/RBS_NeuralNetwork/Training/TestSet"
CATEGORIES = ["ag5au", "ag10au", "ag15au", "ag20au", "ag25au", "ag30au", "ag35au", "ag40au", "ag45au", "ag50au", "ag55au", "ag60au", "ag65au", "ag70au", "ag75au", "ag80au", "ag85au", "ag90au", "ag95au", "ag100au"]
NUM_GENERATED = 100
CHANNELS = 1024

def insert_number(input):
	x = str(input)
	print(x)


def extract_number(input):
	return float(input)


def load_spectrum(path, categorie):
	spectrum = []


	path = os.path.join(DATADIR, categorie)
	s = os.listdir(path)[0]

	f = open(os.path.join(path,s))

	for line in (f.readlines() [-CHANNELS:]):
		x = line.split()
		channel = x[0]
		data = x[1]
		fit = x[2]
		fit_num = extract_number(fit)
		spectrum.append(fit_num)

	return spectrum


def add_noise(spectrum):
	return np.random.poisson(spectrum)


def generate_spectrum(spectrum_data, path):
	fptr = open(path, "w")
	index = 0
	for num in spectrum_data:
		line = str(index) + " x " + str(num) + "\n"
		fptr.write(line)

	return 0


def main():
	for categorie in CATEGORIES:
		spectrum_data = load_spectrum(DATADIR, categorie)
		for i in range(NUM_GENERATED):
			name = "gen" + str(i) + ".dat"
			path = os.path.join(DATADIR, categorie)
			file_path = os.path.join(path, name)

			generate_spectrum(spectrum_data, file_path)
			


	#s = load_spectrum(DATADIR, "ag10au")
	#print(s)
	



if __name__ == "__main__":
    main()
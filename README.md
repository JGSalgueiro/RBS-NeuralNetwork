# Rutherford Backscatering Spectral Classifier

## Rutherford Backscatering
Rutherford backscattering spectrometry (RBS) is an analytical technique used in materials science. Sometimes referred to as high-energy ion scattering (HEIS) spectrometry, RBS is used to determine the structure and composition of materials by measuring the backscattering of a beam of high energy ions (typically protons or alpha particles) impinging on a sample. 

## Spectral Anlysis using Artificial Neural Network(ANN) 
### Description:
- The inverse RBS problem is to determine from the RBS spectrum data the corresponding sample structures. This process is very time consuming and can only be performed by trained specialists. Using Machine learning we can perform an almost instant classification of the data. 
- The development of this project was done using the Tensorflow and Keras software libraries.

### Requirements: 
- python3 (version 3.6 or higher)
- tensorflow (version 2.0 or higher)
- numpy 

## Credits and References
- N.P. Barradas, R.N. Patricio, H.F.R. Pinho, A .Vieira , General artificial neural network for analysis of RBS data, http://projects.ctn.tecnico.ulisboa.pt/Vcorregidor/FCT/3D-NM/26.pdf
- N.P. Barradas, A .Vieira , Artificial neural network analysis of RBS data of Er-implanted sapphire, https://www.sciencedirect.com/science/article/abs/pii/S0168583X00005486

## Experimental condtions for use
- Proton beam with 2MeV of energy;
- Particle detector angle: 140º;
- Angle of incidence: 0º;
- Detector resolution: 30 keV;
- Collected charge: 0.1 microCoulomb;

## Installation
- Install the nessecary libraries in you work environment : tensor-flow (or tensor-flow-gpu), numpy , keras, keras_tunners, pickle, os & matplotlib.
- Depending on your Operating System and work environment of choice, the instalation methods will vary. 

## Usage
### Create_Set
- This scrip generates the data-set used for training the model. The generation method is very simple : Using the Spectra in the /Training/Test_set/<category> folders, generates more by adding poisson noise to them (and normalizing). This method is not the most optimal for all cases but due to the strict experimental conditions it seemed apropriate. More data analysis and management is probably needed in order to obtain better results. 
  
 - To run go to the folder containing the script and write in the terminal : 
 > python3 create_set.py
  
 ### Model_Builder
 - This script will use the Training set to train a model suitable sparse categorical classification. It uses convolutional layers with precision as metric. The model is created with the name "best_model".
  
 - To run go to the folder containing the script and write in the terminal : 
 > python3 model_builder.py
  
### Classifier 
 - To use the trained model for pratical applications, first you'll need to have a folder named /to_classify and there is were the spectra (can be multiple files) to be classified should be placed. Be carefull with the format of the spectra (other formats will not work). A .txt file will be generated with the results.
  
 - To run go to the folder containing the script and write in the terminal : 
 > python3 classifier.py

## Experimental Applications and Results
 -  The experiemental results were presented in the following poster (submited for the [IBA&PIXE - SIMS 2021](http://iba2021.iopconfs.org/home) scientific convention : 
  
 ![alt text](http://url/to/img.png)


## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)


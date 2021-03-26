# Brain Computer Interfaces for Industrial Fluid Flow Control

## Group Members
**Amaani Zuberi**

**Farhan Wadia**

**Jun Sung Park**

**Mrinal Jha** 

**Stephanie DiNunzio**

## Description
This repository consists of all data and software developed for our capstone project.

Programs for collecting EEG data based on covert speech, motor imagery, SSVEP, and eye movements can be found in the `Data Collection Scripts` folder. Follow the instructions in the `Using the Muse Headset for Collecting Raw EEG Signals.pdf` file within that folder to run any of the programs. Note that the MuseLab SDK is not included in this folder and will need to be installed separately.

Raw data files we collected for covert speech model training can be found in the `Nov 2020 Raw Data` and `Feb 2021 Raw Data` folders.

Filtering, time series data outputs, FFT outputs, wavelet transforms, and corresponding graphs were all developed using the `visualization.py` script. Outputs can be found organized in the `Nov 2020 Filtered Data` and `Feb 2021 Filtered Data` folders. 

Model training can be found in the `EEG_Classification_Model_Training.ipynb` notebook. Downloaded model weight files can be found in the `Trained Model Weights` folder. Testing for these models is conducted in the `EEG_Classification_Model_Testing.ipynb` notebook. A breakdown of the results for our final chosen model can be found in the `Final Model Test Results Summary.xlsx` spreadsheet.

Relevant files for implementing the chosen model live in conjunction with a solenoid valve can be found in the `Final Prototype Implementation` folder. Refer to the `Using the Muse Headset for Collecting Raw EEG Signals.pdf` file for further instructions on how to set up the Muse for live EEG streaming, and also refer to `Implementation_README.md`

The `Prototype and Final Design CAD Assemblies` folder contains SOLIDWORKS parts and assemblies showing the mechanical design of our system.

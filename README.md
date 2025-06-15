## Versions
- Python: 3.9.0
- scikit-learn: 1.6.1
- brainflow: 5.6.0
- numpy: 1.23.5
- pandas: 1.4.2

## Processing
- 5-second windows PSD calculated by Welch, with 1-second windows and half a second overlap
- 85:15 training/testing inter-subject data division (1-27 training, and 28-32 testing) (missing cross-validation!)
- normalization using z-score considering base line prior to PSD calculation
- Prior to pre-processing DEAP dataset, they applied (1) bandpass frequency filter from 4.0-45.0 Hz, and (2) averaged to the common reference

## Citation
If you find anything in the paper or repository useful, please consider citing:
```
8-Channel EEG model based on DEAP dataset:
@article{BlancoRos2024,
  title = {Real-time EEG-based emotion recognition for neurohumanities: perspectives from principal component analysis and tree-based algorithms},
  volume = {18},
  ISSN = {1662-5161},
  url = {http://dx.doi.org/10.3389/fnhum.2024.1319574},
  DOI = {10.3389/fnhum.2024.1319574},
  journal = {Frontiers in Human Neuroscience},
  publisher = {Frontiers Media SA},
  author = {Blanco-Ríos,  Miguel Alejandro and Candela-Leal,  Milton Osiel and Orozco-Romo,  Cecilia and Remis-Serna,  Paulina and Vélez-Saboyá,  Carol Stefany and Lozoya-Santos,  Jorge de Jesús and Cebral-Loureda,  Manuel and Ramírez-Moreno,  Mauricio Adolfo},
  year = {2024},
  month = mar 
}

Neurohumanities Lab enhances immersion and learning:
@article{MelladoReyes2024,
author = {Mellado-Reyes, Rebeca and Ortiz, Alexandro and Vélez-Saboyá, Carol S. and Lozoya-Santos, Jorge De-J. and Ramírez-Moreno, Mauricio A. and Cebral-Loureda, Manuel},
title = {Neurohumanities Lab: Physiological Signal Analysis Within an Educational Partially Immersive Environment},
journal = {Evolutionary studies in imaginative culture},
volume = {8},
number = {2},
pages = {383--392},
doi = {https://doi.org/10.70082/esiculture.vi.2312},
year = {2024}
}

8-Channel Neurohumanities Lab data paper:
@article{RomoDeLen2024,
  title = {EEG and Physiological Signals Dataset from Participants during Traditional and Partially Immersive Learning Experiences in Humanities},
  volume = {9},
  ISSN = {2306-5729},
  url = {http://dx.doi.org/10.3390/data9050068},
  DOI = {10.3390/data9050068},
  number = {5},
  journal = {Data},
  publisher = {MDPI AG},
  author = {Romo-De León, Rebeca and Cham-Pérez, Mei Li L. and Elizondo-Villegas, Verónica Andrea and Villarreal-Villarreal, Alejandro and Ortiz-Espinoza, Alexandro Antonio and Vélez-Saboyá, Carol Stefany and Lozoya-Santos, Jorge de Jesús and Cebral-Loureda, Manuel and Ramírez-Moreno, Mauricio A.},
  year = {2024},
  month = may,
  pages = {68}
}
```

## Overview
Within the field of humanities, there is a recognized lack of educational innovation, as there are currently no reported tools available that enable individuals to interact with their environment to create an enhanced learning experience in the humanities. This project proposes a solution to address this gap by integrating technology and promoting the development of teaching methodologies in the humanities, specifically through the incorporation of emotional monitoring during the learning process. 

The main objective of this project is to develop a real-time emotion detection system utilizing EEG signals, which will be interpreted and classified into specific emotions. These emotions will be aligned with the ones proposed by Descartes, including admiration, love, hate, desire, joy, and sadness. By integrating emotional data into the Neurohumanities Lab interactive platform, the aim is to create a comprehensive and immersive learning environment.

## Features
- Real-time emotion detection using EEG signals
- Interpretation and classification of emotions (admiration, love, hate, desire, joy, sadness)
- Integration with the Neurohumanities Lab interactive platform

## Results
The algorithm developed for the Real-Time Emotion Detection achieved better results (92-93% accuracy) than the ones found in consulted literature (88% accuracy). The algorithm focuses on predicting the given emotions, and for further research, it is recommended to consider additional information.

## How to use
1. Download the [DEAP Dataset's](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) preprocessed data in python format using the credentials provided by Queen Mary University of London. You should save these files under a folder named `datos` inside your workspace.
2. Download the .yml file located in this Github, and follow the next steps to create an environment using this file.
    1. Download NeuroEmociones.yml into your desired folder.
    2. Using the Anaconda Prompt, change directory to the one where NeuroEmociones.yml is located using `cd (insert directory)`
    3. Use the command `conda env create -f NeuroEmociones.yml` to create a new environment.
5. Download CargarDatos.ipynb and Model.ipynb into your folder.
6. Run CargarDatos.ipynb to preprocess all the .dat files into an easy to use numpy file. 
7. Once CargarDatos.ipynb is done running, go ahead and create your own ML models, or run Model.ipynb to get several .pkl models to use on your projects. 

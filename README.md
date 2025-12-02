## Citation
If you find anything useful in the journal article or repository, please consider citing:
```
8-Channel EEG model based on DEAP dataset (2024_Frontiers):
@article{BlancoRos2024,
  title = {Real-time EEG-based emotion recognition for neurohumanities: perspectives from principal component analysis and tree-based algorithms},
  volume = {18},
  ISSN = {1662-5161},
  url = {http://dx.doi.org/10.3389/fnhum.2024.1319574},
  DOI = {10.3389/fnhum.2024.1319574},
  journal = {Frontiers in Human Neuroscience},
  publisher = {Frontiers Media SA},
  author = {Blanco-Ríos, Miguel Alejandro and Candela-Leal, Milton Osiel and Orozco-Romo, Cecilia and Remis-Serna, Paulina and Vélez-Saboyá, Carol Stefany and Lozoya-Santos, Jorge de Jesús and Cebral-Loureda, Manuel and Ramírez-Moreno, Mauricio Adolfo},
  year = {2024},
  month = mar 
}
```

> Blanco-Rios MA, Candela-Leal MO, Orozco-Romo C, Remis-Serna P, Velez-Saboya CS, Lozoya-Santos JJ, Cebral-Loureda M, & Ramirez-Moreno MA (2024)<br>
> **Real-time EEG-based Emotion Recognition for Neurohumanities: Perspectives from Principal Component Analysis and Tree-based Algorithms**<br>
> Frontiers in Human Neuroscience, 18, 1319574. https://doi.org/10.3389/fnhum.2024.1319574

## Data
Raw data for `2026_EDUCON` or `2026_Elsevier` is available upon reasonable request.

---

### Requirements
- Python: 3.9.0
- scikit-learn: 1.6.1
- brainflow: 5.6.0
- numpy: 1.23.5
- pandas: 1.4.2
- scipy: 1.12.0
    
### Overview
Within the field of humanities, there is a recognized lack of educational innovation, as there are currently no reported tools available that enable individuals to interact with their environment to create an enhanced learning experience in the humanities. This project proposes a solution to address this gap by integrating technology and promoting the development of teaching methodologies in the humanities, specifically through the incorporation of emotional monitoring during the learning process. 

The main objective of this project is to develop a real-time emotion detection system utilizing EEG signals, which will be interpreted and classified into specific emotions. These emotions will be aligned with the ones proposed by Descartes, including admiration, love, hate, desire, joy, and sadness. By integrating emotional data into the Neurohumanities Lab interactive platform, the aim is to create a comprehensive and immersive learning environment.

### Processing
- 3/5/10-second windows PSD calculated by Welch, with 1-second windows and half a second overlap
- 90:10 training/testing inter-subject data division (1-29 training, and 30-32 testing) (missing cross-validation!)
- engagement, fatigue, excitement, relaxation indices calculation **prior** to normalization. Normalized according to cal indices
- baseline-normalized band powers using: (x - mean(cal))/mean(cal), considering a 3-second baseline prior to the 60-second stimulation
- **Prior** to pre-processing DEAP dataset, they applied (1) bandpass frequency filter from 4.0-45.0 Hz, and (2) averaged to the common reference (this needs to be replicated!)

### Features
- Real-time emotion detection using EEG signals
- Interpretation and classification of emotions (admiration, love, hate, desire, joy, sadness)
- Integration with the Neurohumanities Lab interactive platform

### Results
The algorithm developed for the Real-Time Emotion Detection achieved better results (92-93% accuracy) than the ones found in consulted literature (88% accuracy). The algorithm focuses on predicting the given emotions, and for further research, it is recommended to consider additional information.

### How to use
1. Download the [DEAP Dataset's](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html) preprocessed data in python format using the credentials provided by Queen Mary University of London. You should save these files under a folder named `datos` inside your workspace.
2. Download the .yml file located in this Github, and follow the next steps to create an environment using this file.
    1. Download NeuroEmociones.yml into your desired folder.
    2. Using the Anaconda Prompt, change directory to the one where NeuroEmociones.yml is located using `cd (insert directory)`
    3. Use the command `conda env create -f NeuroEmociones.yml` to create a new environment.
5. Download CargarDatos.ipynb and Model.ipynb into your folder.
6. Run CargarDatos.ipynb to preprocess all the .dat files into an easy to use numpy file. 
7. Once CargarDatos.ipynb is done running, go ahead and create your own ML models, or run Model.ipynb to get several .pkl models to use on your projects. 

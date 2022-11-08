## Overview
**This repository demonstrates the application of the Delaunay-Rips complex by classifying a patient's sleep stage as sleep/wake based on heart rate data.**  
The file [main.ipynb](main.ipynb) guides the user through the process of reproducing the results of our data analysis.   _Note: the whole notebook may take nearly a day to run depending on the processing power of the hardware._  

The data analysis pipeline is inspired by the work of Chung, Hu, and Wu from the paper† [A Persistent Homology Approach to Heart Rate Variability Analysis With an Application to Sleep-Wake Classification](https://www.frontiersin.org/articles/10.3389/fphys.2021.637684/full). With permission from those authors, the processed CGMH heart rate data used for training and validation is found in .csv files in [CGMH_preprocessed_data](CGMH_preprocessed_data).

**†Chung Y-M, Hu C-S, Lo Y-L and Wu H-T (2021) A Persistent Homology Approach to Heart Rate Variability Analysis With an Application to Sleep-Wake Classification. Front. Physiol. 12:637684. doi: 10.3389/fphys.2021.637684**

## Installation
Installation assumes conda is available to be used for environment creation and Python package management. Ensure a working copy of Anaconda (https://www.anaconda.com/) is installed. This repo also depends on the Delaunay-Rips algorithm (https://github.com/amish-mishra/cechmate_DR) for computing the Delaunay-Rips filtration. From the root directory of this repository type the following commands:

```
conda env create -f del_rips_sleep_wake_classification_environment.yml
conda activate del_rips_sleep_wake_classification
python -m ipykernel install --user --name=del_rips_sleep_wake_classification
pip install git+https://github.com/amish-mishra/cechmate_DR.git
jupyter notebook
```

The first command creates a new conda environment and installs into it Python 3.8.5 and the exact Python packages and versions needed to run the code included in this repository.  
The second command activates the newly created environment.  
The third command makes the newly installed environment available for use within Juypyter notebooks.  
The fourth command installs the modified cechmate package with the Delaunay-Rips method.  
The final command opens a Jupyter notebook page in your web browser, which can be used to open and run the included notebooks. Ensure the kernel del_rips_sleep_wake_classification is selected before running the notebook.

## Contents

This repository includes the following key files/directories:

1. [CGMH_preprocessed_data](CGMH_preprocessed_data) - preprocessed 90 patients' time series data for training and 27 patients' time series data for validation, each with 30 sec epochs recorded at 4 Hz (for details, see [A Persistent Homology Approach to Heart Rate Variability Analysis With an Application to Sleep-Wake Classification](https://www.frontiersin.org/articles/10.3389/fphys.2021.637684/full))
2. [persistence_statistics](persistence_statistics) - stores the persistence statistics of the persistence diagrams produced using Delaunay-Rips, Alpha, and Rips on the lag-map-embedded point cloud of the time series data. This directory is populated using the file [persistence_stats.py](persistence_stats.py)
3. [ml_classifiers](ml_classifiers) - stores the SVM classifiers trained on the persistence statistics associated with using Delaunay-Rips, Alpha, and Rips. This directory is populated using [train_ml_classifiers.py](train_ml_classifiers.py)
4. [performance_metrics_tables](performance_metrics_tables) - stores the performance metrics generated by testing the classifiers for each of the 27 patients. This directory is populated using the file [validate_ml_classifiers.py](validate_ml_classifiers.py)
5. [main.ipynb](main.ipynb) - this is the main notebook that carries out the data analysis step-by-step by calling the appropriate python scripts

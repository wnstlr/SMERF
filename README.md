# SMERF (**S**imulated **M**od**E**l **R**easoning Evaluation **F**ramework) for Saliency Methods

This repository contains the code for the following [paper](https://arxiv.org/abs/2105.06506).

## Requirements

The code was tested with Python 3.6.10. Other requirements can be installed via running ```pip install -r requirements.txt```

Core components among these requirements are packages used to run the saliency methods. 
Refer to these repositories for further instructions on how they are used. 
* [iNNvestigate](https://github.com/albermax/innvestigate)
* [DeepLIFT](https://github.com/kundajelab/deeplift)
* [SHAP](https://github.com/slundberg/shap)

## Run

First create subdirectories required for saving intermediary data and results.

```mkdir outputs outputs/cache outputs/plots```

`data` directory is used to save `.npz` files for the generated datasets.
`outputs/cache` directory is used to store all intermediary/final outputs from SMERF, e.g. trained models, saliency outputs, evalauted metric values, etc.
`outputs/plots` is used to store all plots generated in the process. 

Run the script file in `scripts` directory:

```cd scripts; sh run.sh``` 

This will run the whole pipeline of SMERF for all model reasoning specified in the paper. It will generating a dataset with specified ground-truth, train a model, run saliency methods, evaluate them based on different metrics, and save the outputs. 

Each model reasoning presented in the paper is labeled with a specific number throughout the whole code as the following.

- Simple-FR: 1.11
- Simple-NR: 2.11
- Complex-FR: 1.20
- Complex-CR1: 3.71
- Complex-CR1: 3.72
- Complex-CR1: 3.73
- Complex-CR1: 3.74

Any outputs generated in SMERF for each of these cases will have these numbers included in the filename. The script file also can be modified to only run specific experiments by setting `--exp` value explicitly to one of these numbers.

Once the script file is run, the plots presented in the paper can be reproduced from Jupyter notebooks in the `notebooks` directory.

## Link for data and results used for the paper

The specific files in `data` and `outputs` directories that were used for our results in the paper are available [here](https://drive.google.com/drive/folders/1KzC3QrPYAri4Uyd6HVfoGbPRi3jdaGDO?usp=sharing). 

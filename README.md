# SMERF (**S**imulated **M**od**E**l **R**easoning Evaluation **F**ramework) for Saliency Methods

This repository contains code release for SMERF, a ground-truth-based evaluation framework for saliency methods.

## Requirements

The repository was developed and tested with Python 3.6.10. Other requirements can be installed via running ```pip install -r requirements.txt```. 

Core components among these requirements are packages used to run the saliency methods. Refer to these repositories for further instructions on how they are installed and used. 
* [iNNvestigate](https://github.com/albermax/innvestigate)
* [DeepLIFT](https://github.com/kundajelab/deeplift)
* [SHAP](https://github.com/slundberg/shap)

## Datasets

`data` directory is used to save `textbox_{EXP_NO}.npz` files for the generated datasets. Replace `EXP_NO` with the experiment number specific to each model reasoning setting as shown below. 

- Simple-FR: 1.11
- Simple-NR: 2.11
- Complex-FR: 1.20
- Complex-CR1: 3.71
- Complex-CR1: 3.72
- Complex-CR1: 3.73
- Complex-CR1: 3.74

**Note that throughout the entire code and the set of outputs, parts related to specific setting are labeled with these specific experiment numbers.**

Individual elements of the dataset can be accessed via the following keys:

- `'x_train'`: training data of dimensions (N x 64 x 64 x 3) composed of  N images (`numpy.ndarray`)
- `'y_train'`: training labels of dimensions (N,) composed of N binary labels (`numpy.ndarray`)
- `'x_test'` : test data of dimensions (N x 64 x 64 x 3) composed of N images (`numpy.ndarray`)
- `'y_test'` : test labels of dimensions (N,) composed of N binary labels (`numpy.ndarray`)
- `'train_primary'`: array of coordinates of primary objects (i.e. objects that shoud be highlighted) in each image of the training data (`numpy.ndarray`) 
- `'train_secondary'`: list of coordinates of secondary objects (i.e. objects that should *not* be highlighted) in each image of the training data (`numpy.ndarray`)
- `'test_primary'`: list of coordinates of primary objects in each image of the test data (`numpy.ndarray`)
- `'test_secondary'`: list of coordinates of secondary objects in each image of the test data (`numpy.ndarray`)

Several examples of accessing each elements of the dataset and visualizing them is demonstrated in a Jupyter notebook: `notebooks/extra figures.ipynb`. Download the data files from [here](https://bit.ly/37DpMU3). 

## Running the benchmarking experiments

##### Create subdirectories required for saving intermediary data and results:

```mkdir outputs outputs/cache outputs/plots```

- `outputs/cache` directory is used to store all intermediary/final outputs from SMERF, e.g. trained models, saliency outputs, evalauted metric values, etc.
- `outputs/plots` is used to store all plots generated in the process. 

##### To individually run each model reasoning settings:

```cd scripts; python run_experiments.py --exp {EXP_NO}```

Replace EXP_NO with the experiment number specific to each model reasoning setting as shown earlier. 


##### To run the entire set of model reasoning settings:

Run the script file in `scripts` directory: `cd scripts; bash run.sh`

This will run the whole pipeline of SMERF for all model reasoning specified in the paper. It will generating a dataset with specified ground-truth, train a model, run saliency methods, evaluate them based on different metrics, and save the outputs. 

Any outputs generated in SMERF for each of these cases will have the corresponding `EXP_NO` included in the filename.

##### Notebooks for Plotting 

After the script files are run and the metric values are all computed, the plots presented in the paper can be reproduced from Jupyter notebooks in the `notebooks` directory.

- `IOU_AFL_MAFL_plots.ipynb`: plots for the IOU, AFL, and MAFL metrics.
- `failures.ipynb`: plots for bucket-wise detials on failure cases (i.e low metric values)
- `id_test.ipynb`: side-by-side comparison of feature attributions obtained from different model reasoning on the same input.
- `obj_added.ipynb`: increasing number of objects added to the image and observing the changes of metric values.
- `extra figures.ipynb`: extra figures from the Appendix in the paper.

## Link for data and results used for the paper

The specific files in `data` and `outputs` directories that were used to generate our results in the paper are available [here](https://drive.google.com/drive/folders/1KzC3QrPYAri4Uyd6HVfoGbPRi3jdaGDO?usp=sharing). 

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

Several examples of accessing each elements of the dataset and visualizing them is demonstrated in a Jupyter notebook: `notebooks/extra figures.ipynb`. Download the specific data files used for the experiments in the paper from [here](https://drive.google.com/file/d/1ShdZDTPM1r7dVVuXUI3TushLArSHzDbl/view?usp=sharing), unzip, and place the content in the `data` directory.

#### Generating new datasets

To generate new datasets from scratch, simply run the code without any files in the ```data``` directory. The code will check for existing files to refer to, otherwise generate new ones according to each setting. 

#### Generating dataset with natural backgrounds

We used [Places365 dataset](http://places2.csail.mit.edu/download.html) for the backgrounds. Specifically, download [small-size validation images](http://data.csail.mit.edu/places/places365/val_256.tar) and [metadata](http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar), then place the unzipped folders under the ```data``` directory. Then run 

```cd scripts; python get_places365.py```

to create a metadata pickle file (`data/places_img_file.pkl`) that will be used for generating the image files and running the experiments.

Specific set of images used in our experiment for natural background setting is available [here](https://drive.google.com/file/d/1vppHFKI-4QCrbn_xJRiPTRFgRP2siYWu/view?usp=sharing), under the same format as the original dataset.

## Running the benchmarking experiments

#### Create subdirectories required for saving intermediary data and results:

```mkdir outputs outputs/cache outputs/plots```

- `outputs/cache` directory is used to store all intermediary/final outputs from SMERF, e.g. trained models, saliency outputs, evaluated metric values, etc.
- `outputs/plots` is used to store all plots generated in the process. 

#### To individually run each model reasoning settings:

```cd scripts; python run_experiments.py --exp {EXP_NO} --bg {BACKGROUND_NATURAL} --model_type {MODEL_TYPE} --ep {EPOCH_NO} --lr {LR}```

Replace EXP_NO with the experiment number specific to each model reasoning setting as shown earlier. 

Replace BACKGROUND_NATURAL with either 0 or 1, for black background or natural background respectively (default is black background).

Replace MODEL_TYPE with 0 (default), 1, or 2, each corresponding to simple CNN, VGG16, and AlexNet for the base model.

Replace EPOCH_NO with maximum epoch number for training.

Replace LR with learning rate for training.

#### To run the entire set of model reasoning settings:

Run the script file in `scripts` directory: `cd scripts; bash run.sh`.
Modify `--bg` arguments in the script file to 1 to run experiments with natural images.

This will run the whole pipeline of SMERF for all model reasoning specified in the paper. It will generating a dataset with specified ground-truth, train a model, run saliency methods, evaluate them based on different metrics, and save the outputs. 

Any outputs generated in SMERF for each of these cases will have the corresponding `EXP_NO` included in the filename.

#### Notebooks for Plotting 

After the script files are run and the metric values are all computed, the plots presented (similar to the ones) in the paper can be reproduced from several Jupyter notebooks under the `notebooks` directory.

- `IOU_AFL_MAFL_plots.ipynb`: plots for the IOU, AFL, and MAFL metrics.
- `IOU-AFL-architecture-plots.ipynb`: plots for the IOU, AFL for different architectures.
- `comparison-backgrounds.ipynb`: plots comparing cases with different background (black vs real)
- `failures.ipynb`: plots for bucket-wise detials on failure cases (i.e low metric values)
- `id_test.ipynb`: side-by-side comparison of feature attributions obtained from different model reasoning on the same input.
- `obj_added.ipynb`: increasing number of objects added to the image and observing the changes of metric values.
- `extra figures.ipynb`: extra figures from the Appendix in the paper.

#### Adding new experimental setup to the pipeline

To add new experimental setups, new datasets based on new model reasoning should be generated. Copy `smerf/simple_fr.py` to a new file and modify the code within to generate new datasets with new type of reasoning and features. The essence is that the output of the function `generate_textbox_data()` in the new file should inlcude training/testing data (both images and labels), and the coordinates of the primary and secondary objects in each image in the format provided. This information will be used for computing the evaluation metrics in the end. 

Models can be modified from `smerf/models.py` file. 

#### Adding new saliency methods to the pipeline

The model is a CNN compiled with `keras==2.2.4, tensorflow==1.12.0`. Refer to `smerf/models.py` file for the details of the model. Therefore, new saliency methods added to the pipeline should be implemented for this model.

To add new saliency methods, we recommend writing a helper function that takes in the data and the model, computes the attribution values for the data, and concatenates those values to the existing set of results obtained from other methods. Refer to NOTEs under `smerf/explanations.py` for details on where and how new methods should be added. 

## Link for results used in the paper

The specific files in `outputs` directories that were used to generate our results in the paper are available [here](https://drive.google.com/drive/folders/1E__OIsOqhV6wSkuRORLeaeFQhVZhohKP?usp=sharing). Unzip and place the contents in the respective directory to reproduce the results presented in the paper. There are three files: `outputs.zip` for the simple CNN, `outputs_alex.zip` for AlexNet, `outputs_vgg.zip` for VGG16, and `outputs_baseball.zip` for the simple CNN using real background images.

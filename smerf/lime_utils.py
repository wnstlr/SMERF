import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import tensorflow as tf
import keras.backend as K
import numpy as np
from smerf.models import *

# Uses LIME library to obtain feature attributions
def lime_run(model, x_sample, y_sample, x_train, exp_no):
    n, h, w, c = x_sample.shape
    output = np.zeros((n, 1, h, w, c))
    explainer = lime_image.LimeImageExplainer(verbose=False)
    for n_i in range(n):
        explanation = explainer.explain_instance(x_sample[n_i].astype('double'), 
                                                 model.predict, 
                                                 top_labels=2,
                                                 hide_color=(0,0,0), 
                                                 num_samples=1000,  
                                                 segmentation_fn=SegmentationAlgorithm('quickshift', 
                                                                                       kernel_size=4,
                                                                                       max_dist=10, 
                                                                                       ratio=0.2)
                                                 )
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
        heatmap = scale(heatmap)
        output[n_i, 0, :, :, 0] = heatmap
        output[n_i, 0, :, :, 1] = heatmap
        output[n_i, 0, :, :, 2] = heatmap
    return output 

def scale(x):
    if x.max() - x.min() != 0:
        return (x - x.min()) / (x.max() - x.min())
    return x
import numpy as np
import imp
textcolorutils = imp.load_source('textcolor_utils', '../smerf/textcolor_utils.py')
import innvestigate
import keras
import keras.backend as K
import tensorflow as tf
import cv2
import pickle
import os
from tqdm import tqdm


# NOTE Helper functions for saliency methods that are not supported by iNNvestigate libary
#      We recommend specifying these methods on a separate file to avoid clutter.
from .grad_cam_utils import *
from .shap_utils import *
from .lime_utils import *

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]

def run_methods(model, 
                x_data, 
                y_data, 
                x_train, 
                no_images=10, 
                exp_no=0, 
                load=True, 
                f_name=None, 
                directory='../outputs/cache', 
                split=None,
                model_type=0):
    """
    Given a trained model, run saliency methods and return the results.

    :param model: keras sequntial model to be explained
    :param x_data: (N, H, W, C) image dataset
    :param y_data: (N,) image labels
    :param no_images: number of sample images to run methods on (n)
    :param exp_no: experiment number for reference
    :param load: if True, load from a cached results
    :param directory: directory to save or load the results
    :param split: if not None, specifies the bucket index the methods are run on 
    :param model_type: 0 (default) for simple CNNs. 1 for VGG16. 2 for ResNet50.

    :return (result, method, text, idx) tuple where result is (no_images, M, H, C, W) saliency output 
    for M methods and no_images images; method is the list of configs for the saliency methods; 
    text is the prediction information on the images; idx is the original index of the sampled images from the dataset
    """
    model_in = keras.models.Model(inputs=model.inputs,
                       outputs=model.outputs)
    if split is None:
        result_name = os.path.join(directory, 'result_%0.2f.pkl'%exp_no)
        idx_name = os.path.join(directory, 'idx_%0.2f.pkl'%exp_no)
        methods_name = os.path.join(directory, 'methods_%0.2f.pkl'%exp_no)
        text_name = os.path.join(directory, 'text_%0.2f.pkl'%exp_no)
    else:
        result_name = os.path.join(directory, 'result_%0.2f_%d.pkl'%(exp_no, split))
        idx_name = os.path.join(directory, 'idx_%0.2f_%d.pkl'%(exp_no, split))
        methods_name = os.path.join(directory, 'methods_%0.2f_%d.pkl'%(exp_no, split))
        text_name = os.path.join(directory, 'text_%0.2f_%d.pkl'%(exp_no, split)) 

    if os.path.exists(result_name) and os.path.exists(idx_name) and load==True:
        loaded = True
        print('loading results from cache in {}'.format(directory))
        result = pickle.load(open(result_name, 'rb'))
        idx = pickle.load(open(idx_name, 'rb'))
        methods = pickle.load(open(methods_name, 'rb'))
        text = pickle.load(open(text_name, 'rb'))
    else:
        loaded = False
        if f_name is not None:
            result_name = os.path.join(directory, f_name+'.pkl')
            idx_name = os.path.join(directory, f_name+'_idx.pkl')
            methods_name = os.path.join(directory, f_name+'_methods.pkl')
            text_name = os.path.join(directory, f_name+'_text.pkl')
        print('cache not found')
        noise_scale = 0.1
        input_range = (0,1)
        
        ## NOTE Methods defined here is using the iNNvestigate library.
        ##      Other methods should be added manually via separately defined helper functions at the end.
        methods = [
            # NAME                    OPT.PARAMS                POSTPROC FXN                TITLE
            # Show input.
            ("input",                 {},                       textcolorutils.identity,         "Input"),

            # Function
            ("gradient",              {"postprocess": "abs"},   textcolorutils.graymap,       "Gradient"),
            ("smoothgrad",            {"augment_by_n": 64,
                                    "noise_scale": noise_scale,
                                    "postprocess": "square"},textcolorutils.graymap,       "SmoothGrad"),

            # Signal
            ("deconvnet",             {},                       textcolorutils.bk_proj,       "Deconvnet"),
            ("guided_backprop",       {},                       textcolorutils.bk_proj,       "Guided Backprop",),

            # Interaction
            ("deep_taylor.bounded",   {"low": input_range[0],
                                    "high": input_range[1]}, textcolorutils.heatmap,       "DeepTaylor"),
            ("input_t_gradient",      {},                       textcolorutils.heatmap,       "Input * Gradient"), 
            ("integrated_gradients",  {"reference_inputs": input_range[0], "steps": 64},            textcolorutils.heatmap,       "Integrated Gradients"),
            ("lrp.z",                 {},                       textcolorutils.heatmap,       "LRP-Z"),
            ("lrp.epsilon",           {"epsilon": 1},           textcolorutils.heatmap,       "LRP-Epsilon"),
            ("lrp.sequential_preset_a_flat",{"epsilon": 1},     textcolorutils.heatmap,       "LRP-PresetAFlat"),
            ("lrp.sequential_preset_b_flat",{"epsilon": 1},     textcolorutils.heatmap,       "LRP-PresetBFlat"),
            ("deep_lift.wrapper", {"nonlinear_mode":"reveal_cancel", "reference_inputs": 0, "verbose": 0}, textcolorutils.heatmap, "DeepLIFT-RevealCancel"),
            ("deep_lift.wrapper", {"nonlinear_mode":"rescale", "reference_inputs": 0, "verbose":0}, textcolorutils.heatmap, "DeepLIFT-Rescale"),
        ]

        # Create analyzers.
        analyzers = []
        for method in methods:
            try:
                analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                        model_in, # model without softmax output
                                                        **method[1])      # optional analysis parameters
            except innvestigate.NotAnalyzeableModelException:
                # Not all methods work with all models.
                analyzer = None
            analyzers.append(analyzer)

        # Run the saliency methods
        text = []
        label_to_class_name = {0: 'Neg', 1: 'Pos'}
        color_conversion = None
        channels_first = keras.backend.image_data_format() == "channels_first"

        # random set of images to test
        np.random.seed(1)
        idx = np.random.choice(len(x_data), no_images, replace=False) # index of the test data selected

        images = x_data[idx]
        labels = y_data[idx]
        h, w, c = images[0].shape
        result = np.zeros((len(images), len(analyzers), h, w, c))

        # Run methods on batch
        for aidx, analyzer in enumerate(tqdm(analyzers)):
            if methods[aidx][0] == "input":
                    # Do not analyze, but keep not preprocessed input.
                    a = images
            elif analyzer:
                if model_type == 1 or model_type ==2: # NOTE for fine-tuned model, deep lift does not support maxpooling2d
                    if 'deep_lift' not in methods[aidx][0]:
                        # Analyze.
                        a = analyzer.analyze(images)
                        # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                        a = textcolorutils.postprocess(a, color_conversion, channels_first)
                        # Apply analysis postprocessing, e.g., creating a heatmap.
                        a = methods[aidx][2](a)
                    else:
                        print('deepLift does not support specific layers')
                        a = np.zeros_like(images)
                else:
                    # Analyze.
                    a = analyzer.analyze(images)
                    # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                    a = textcolorutils.postprocess(a, color_conversion, channels_first)
                    # Apply analysis postprocessing, e.g., creating a heatmap.
                    a = methods[aidx][2](a)
            else:
                a = np.zeros_like(images)
            # Store the analysis.
            result[:, aidx] = a

        # Predict final activations, probabilites, and label.
        presm = model.predict(images)
        prob = softmax(presm)
        y_hat = prob.argmax(axis=1)

        for i, y in enumerate(labels):
            # Save prediction info:
            text.append(("%s" % label_to_class_name[y],    # ground truth label
                        "%.2f" % presm.max(axis=1)[i],             # pre-softmax logits
                        "%.2f" % prob.max(axis=1)[i],              # probabilistic softmax output  
                        "%s" % label_to_class_name[y_hat[i]] # predicted label
                        ))

        ####### NOTE Add additional methods to run below (those that are not supported 
        #######      in the iNNvestigate library)

        # Add Grad-CAM
        print(' Running Grad-CAM')
        result, methods = add_grad_cam(result, methods, model, images, exp_no, directory, model_type)

        # Add SHAP
        print(' Running SHAP')
        result, methods = add_shap(result, methods, model, images, labels, x_train, exp_no, model_type)
        
        # # Add LIME
        # print(' Running LIME')
        # result, methods = add_lime(result, methods, model, images, labels, x_train, exp_no)

        # random baseline
        print(' Running random')
        random_results = np.random.random((len(images), 1, h, w, c))
        methods.append(('random',{}, textcolorutils.heatmap, "Random"))

        # edge detector
        print(' Running edge detection')
        edge_results = np.zeros((len(images), 1, h, w, c))
        for i, x in enumerate(images):
            ed = cv2.Sobel(x, cv2.CV_64F,1,0,ksize=5)
            ed = (ed - np.min(ed)) / (np.max(ed) - np.min(ed))
            edge_results[i,0] = ed
        methods.append(('edge', {}, textcolorutils.heatmap, "Edge-detection"))
        result = np.concatenate((result, random_results, edge_results), axis=1)
        
        ###### Adding new methods should be completed above ######

        if not loaded:
            # Save results
            pickle.dump(result, open(result_name, 'wb'))
            pickle.dump(idx, open(idx_name, 'wb'))
            pickle.dump(methods, open(methods_name, 'wb'))
            pickle.dump(text, open(text_name, 'wb'))

        ## saliency results: size (no_samples, methods, img_size)
        print(' No-images: %d \t No-methods: %d finished.'%(result.shape[0], result.shape[1]))

    return result, methods, text, idx

#### Helper functions for additional explanation methods to include ####
#### NOTE Add additional explanation methods below
####      Define methods that will take in the result matrix containing outputs from other methods
####      and add the new method's result to return. 
####      The result matrix has a shape (no_samples, no_methods, h, w, c), so the output of the new methods
####      defined below should respect this format and be concatenated to the existing result matrix
####      along axis=1 (no_methods). 
####      Below shows examples of adding two new methods (GradCAM and DeepSHAP) and basic steps that 
####      should be followed within these functions, what should be returned in the end.

# Function for adding GradCAM results to the existing results.
def add_grad_cam(result, methods, model, images, exp_no, directory, model_type):
    
    # compute attributions
    if model_type == -1: # adversarial
        model_name = 'w-adv-%0.2f.pt'%exp_no
    elif model_type == 0:
        model_name = 'w%0.2f.pt'%exp_no
    elif model_type == 1:
        model_name = 'w_vgg%0.2f.pt'%exp_no
    elif model_type == 2:
        model_name = 'w_alex%0.2f.pt'%exp_no
    else:
        raise ValueError('model_type not supported')
    c, h, g = grad_cam_run(model, images, os.path.join(directory, model_name), exp_no, model_type)
    
    # add the new results to the existing results
    added_result = np.expand_dims(h, 1)
    result = np.concatenate((result, added_result), axis=1)
    
    # add the new method information to the existing method information
    methods.append(('grad-cam', {}, textcolorutils.graymap, "Grad-CAM"))
    
    # return both result and method information
    return result, methods

# Function for adding DeepSHAP results to the existing results.
def add_shap(result, methods, model, images, labels, x_train, exp_no, model_type):
    
    # compute attributions
    output = shap_run(model, images, labels, x_train, exp_no, model_type)
    h, w, c = images[0].shape
    assert(output.shape == (images.shape[0], 1, h, w, c))
    
    # add the new results to the existing results
    result = np.concatenate((result, output), axis=1)
    
    # add the new method information to the existing method information
    methods.append(('deep-shap', {}, textcolorutils.graymap, "DeepSHAP"))
    
    # return both result and method information
    return result, methods

def add_lime(result, methods, model, images, labels, x_train, exp_no):
    
    # compute attributions
    output = lime_run(model, images, labels, x_train, exp_no)
    h, w, c = images[0].shape
    assert(output.shape == (images.shape[0], 1, h, w, c))
    
    # add the new results to the existing results
    result = np.concatenate((result, output), axis=1)
    
    # add the new method information to the existing method information
    methods.append(('lime', {}, textcolorutils.graymap, "LIME"))
    
    # return both result and method information
    return result, methods

# New method to be added to the pipeline
def add_your_new_method(result, methods, model, images, **kwargs):
    # TODO fill in the code for your new method to be added.
    return result, methods

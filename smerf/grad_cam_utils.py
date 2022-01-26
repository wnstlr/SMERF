"""
Source code adapted from https://github.com/wawaku/grad-cam-keras
"""

from keras.preprocessing import image
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import cv2
import os, gc

from .models import TextBoxCNN as TextBoxCNN
from .models import TextBoxCNN_adv as TextBoxCNN_adv
from .models import VGG16_model as VGG16_model
from .models import AlexNet_model as AlexNet_model

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def compile_saliency_function(model, activation_layer='block5_pool'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name, model_file, exp_no, model_type):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        if model_type == 0:
            if exp_no >= 3.5 or exp_no == 1.2:
                new_model = TextBoxCNN_adv().model
            else:
                new_model = TextBoxCNN().model
        elif model_type == 1:
            new_model = VGG16_model().model
        elif model_type == 2:
            new_model = AlexNet_model().model
        else:
            raise ValueError('model_type must be 0, 1, or 2')
        new_model.load_weights(model_file)
    return new_model


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # avg
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # create heat map
    cam = cv2.resize(cam, (x.shape[1], x.shape[2]), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap

def grad_cam_run(model, x_sample, model_file, exp_no, model_type):
    last_conv_layer_name = [x for x in model.layers if type(x) == keras.layers.convolutional.Conv2D][-1].name
    cam_imgs = np.zeros(x_sample.shape)
    heat_maps = np.zeros(x_sample.shape)
    grad_cam_imgs = np.zeros(x_sample.shape)
    for i in range(x_sample.shape[0]):
        img = x_sample[i][None,:,:,:]
        predictions = model.predict(img)
        pred_class = predictions.argmax(axis=1)[0]
        cam_image, heat_map = grad_cam(model, img, pred_class, last_conv_layer_name)

        # guided grad_cam img 
        register_gradient()
        guided_model = modify_backprop(model, 'GuidedBackProp', model_file, exp_no, model_type)
        guided_model_name = [x for x in guided_model.layers if type(x) == keras.layers.convolutional.Conv2D][-1].name
        saliency_fn = compile_saliency_function(guided_model, activation_layer=guided_model_name)
        saliency = saliency_fn([img, 0])
        grad_cam_img = saliency[0] * heat_map[..., np.newaxis]

        if np.max(grad_cam_img) - np.min(grad_cam_img) != 0:
            grad_cam_img = (grad_cam_img - np.min(grad_cam_img)) / (np.max(grad_cam_img) - np.min(grad_cam_img))

        cam_imgs[i] = cam_image
        heat_maps[i] = np.repeat(heat_map[:, :, np.newaxis], 3, axis=2)
        grad_cam_imgs[i] = grad_cam_img[0]
    del guided_model
    gc.collect()
    return cam_imgs, heat_maps, grad_cam_imgs

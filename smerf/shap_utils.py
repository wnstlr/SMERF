import shap
import tensorflow as tf
import keras.backend as K
import numpy as np
from smerf.models import *
import gc

# Uses SHAP library to obtain feature attributions
def shap_run(model, x_sample, y_sample, x_train, exp_no, model_type):
    ## NOTE due to complications in keras and TF versions (this code works only in TF1), 
    ## a separate model should be redefined on a separate session where SHAP is run. 
    model.save_weights('/tmp/model.pt')
    background = x_train[np.random.choice(x_train.shape[0], 1000, replace=False)]
    n, h, w, c = x_sample.shape
    output = np.zeros((n, 1, h, w, c))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # redefine the model
        if model_type == 0:
            if exp_no >= 3.5 or exp_no==1.2:
                model_obj = TextBoxCNN_adv(lr=0.0001, max_epoch=10)
            else:
                model_obj = TextBoxCNN(lr=0.0001, max_epoch=10)
        elif model_type == 1:
            model_obj = VGG16_model()
        elif model_type == 2:
            model_obj = AlexNet_model()
        else:
            raise ValueError('model_type must be 0, 1, or 2')
        model_obj.model.load_weights('/tmp/model.pt')
        model_sess = model_obj.model
        # DeepSHAP
        e = shap.DeepExplainer(model_sess, background)
        shap_vals_deep = e.shap_values(x_sample)
        #shap_vals_deep = np.array([shap_vals_deep[y_sample[i]][i] for i in range(n)])
        shap_vals_deep = np.array([scale(shap_vals_deep[y_sample[i]][i]) for i in range(n)])
        shap_vals_deep = np.max(shap_vals_deep, axis=3)
        shap_vals_deep = np.expand_dims(shap_vals_deep, 3)
        shap_vals_deep = np.concatenate((shap_vals_deep, shap_vals_deep, shap_vals_deep), axis=3)
        output[:, 0, :, :, :] = shap_vals_deep
    return output

def scale(x):
    if x.max() - x.min() != 0:
        return (x - x.min()) / (x.max() - x.min())
    return x

import shap
import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
import smerf
from smerf.models import *
import pickle
import argparse
from smerf.eval import *

DATA_DIR = '../data'

# Standalone function for running SHAP and adding it to the results
def insert_shap_results(x_train, x_sample, y_sample, model_type, exp_no, cache_dir): 
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
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
            model_fname = 'w%0.2f.pt'%exp_no
        elif model_type == 1:
            model_obj = VGG16_model()
            model_fname = 'w_vgg%0.2f.pt'%exp_no
        elif model_type == 2:
            model_obj = AlexNet_model()
            model_fname = 'w_alex%0.2f.pt'%exp_no
        else:
            raise ValueError('model_type must be 0, 1, or 2')
        model_obj.model.load_weights(os.path.join(cache_dir, model_fname))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=float, help='experiment number. 1-FR, 2-NR, 3-CR')
    parser.add_argument('--model_type', type=int, default=0, help='set this to 0 for simple CNN, 1 for vgg16, 2 for AlexNet for the base model')
    parser.add_argument('--cache_dir', type=str, default='../outputs/cache', help='')
    args = parser.parse_args()
    exp_no = args.exp
    model_type = args.model_type
    cache_dir = args.cache_dir
     
    # files to overwrite
    idx = pickle.load(open(os.path.join(cache_dir, 'idx_%0.2f.pkl'%exp_no), 'rb'))
    results = pickle.load(open(os.path.join(cache_dir, 'result_%0.2f.pkl'%exp_no), 'rb'))
    methods = pickle.load(open(os.path.join(cache_dir, 'methods_%0.2f.pkl'%exp_no), 'rb'))
    text_name = pickle.load(open(os.path.join(cache_dir, 'text_%0.2f.pkl'%exp_no), 'rb'))

    assert(results.shape[1] <= 18)
    
    # load data 
    data = np.load(open(os.path.join(DATA_DIR, 'textbox_%0.2f.npz'%exp_no), 'rb'))
    x_train = data['x_train']
    x_test = data['x_test']
    y_test = data['y_test']
    x_sample = x_test[idx]
    y_sample = y_test[idx]

    # compute shap
    output = insert_shap_results(x_train, x_sample, y_sample, model_type, exp_no, cache_dir)
    
    # add the new method information to the existing method information at the specified location
    results = np.concatenate((results[:,:-2], output, results[:,-2:]), axis=1)
    methods = methods[:-2] + [('deep-shap', {}, textcolorutils.graymap, "DeepSHAP")] + methods[-2:]
    
    # overwrite the cache results
    pickle.dump(results, open(os.path.join(cache_dir, 'result_%0.2f.pkl'%exp_no), 'wb'))
    pickle.dump(methods, open(os.path.join(cache_dir, 'methods_%0.2f.pkl'%exp_no), 'wb'))
    
    print('added shap results %s %s'%(exp_no, model_type))
    

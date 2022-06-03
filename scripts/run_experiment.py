import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../')
import smerf
from smerf.models import *
from smerf.textbox_data import *
from smerf.eval import *
import smerf.explanations as saliency
import argparse

# Directory to save the data
DATA_DIR = '../data'

# Directory to save intermediary/final results
CACHE_DIR = '../outputs/cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
    
# Directory to save plots
PLOT_DIR = '../outputs/plots'
if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)

def main(args):
    exp_no = args.exp # experiment number
    adv_train = args.adv #whether to train with adversarial examples
    print('EXP_NO = %f'%exp_no)
    lr = args.lr # learning rate used to train the model
    epoch = args.ep # number of maximum epochs to train the model
    random_bg = args.bg # whether to use random background
    model_type = args.model_type # whether to use advanced models
    if adv_train:
        assert(model_type==-1)
    if model_type == -1:
        assert(adv_train)
        model_name = 'w-adv-%0.2f.pt'%exp_no #adversarially trained model
    elif model_type == 0:
        model_name = 'w%0.2f.pt'%exp_no # model file to save/load
    elif model_type == 1:
        model_name = 'w_vgg%0.2f.pt'%exp_no 
    elif model_type == 2:
        model_name = 'w_alex%0.2f.pt'%exp_no 
    else: 
        raise ValueError('Invalid model type')

    ### NOTE Below defines the hyperparameters for each experiment setups. 
    ### These should be stated explicitly for every new experiment setups added.
    if exp_no == 1.11: ## Simple-FR
        import smerf.simple_fr as textbox_exp # import textbox data generator method define separately
        no_data = 2000 # number of training data points per bucket 
        no_test_per_split = 500 # number of test data points per bucket
        no_split = 12 # number of buckets
        
        # define the bucket IDs and names: information about features used
        split_names = dict()
        patch = [0,1]
        text = ['None','A','B']
        switch = [0,1]
        count = 0
        for p in patch:
            for t in text:
                for s in switch:
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s,p,t)
                    count += 1
        # make sure that the number of buckets is set up correctly.
        assert(no_split == count)

    elif exp_no == 2.11: ## Simple-NR
        import smerf.simple_nr as textbox_exp
        no_data = 5000
        no_test_per_split = 500
        no_split = 8
        split_names = dict()
        switch = [0,1]
        patch = [0,1]
        text = ['A','B']
        count = 0
        for s in switch:
            for p in patch:
                for t in text:
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s, p, t)
                    count += 1
        assert(no_split==count)

    elif exp_no == 1.2: ## Complex-FR
        import smerf.complex_fr as textbox_exp
        no_data = 2000
        no_test_per_split = 500
        no_split = 12
        split_names = dict()
        patch = [0,1]
        text = ['None','A','B']
        switch = [0,1]
        count = 0
        for p in patch:
            for t in text:
                for s in switch:
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s,p,t)
                    count += 1
        assert(no_split == count)
        
    elif exp_no == 3.71: # Complex-CR1
        import smerf.complex_cr1 as textbox_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        split_names = dict()
        switch = [0,2]
        patch = [0,1]
        text = ['None','A','B']
        count = 0
        for s in switch:
            for p in patch:
                for t in text:
                    if t == 'None' and s == 0: # undecided regions
                        continue
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s, p, t)
                    count += 1
        assert(no_split == count)

    elif exp_no == 3.72: # Complex-CR2
        import smerf.complex_cr2 as textbox_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        split_names = dict()
        switch = [0,2]
        patch = [0,1]
        text = ['None','A','B']
        count = 0
        for s in switch:
            for p in patch:
                for t in text:
                    if t == 'None' and p == 0: # undecided regions
                        continue
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s, p, t)
                    count += 1
        assert(no_split == count)

    elif exp_no == 3.73: # Complex-CR3
        import smerf.complex_cr3 as textbox_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        split_names = dict()
        switch = [0,2]
        patch = [0,1]
        text = ['None','A','B']
        count = 0
        for s in switch:
            for p in patch:
                for t in text:
                    if t == 'None' and s == 2: # undecided regions
                        continue
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s, p, t)
                    count += 1
        assert(no_split == count)

    elif exp_no == 3.74: # Complex-CR4
        import smerf.complex_cr4 as textbox_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        split_names = dict()
        switch = [0,2]
        patch = [0,1]
        text = ['None','A','B']
        count = 0
        for s in switch:
            for p in patch:
                for t in text:
                    if t == 'None' and p == 1: # undecided regions
                        continue
                    split_names[count] = 'Switch=%d / Patch=%d / Character=%s'%(s, p, t)
                    count += 1
        assert(no_split == count)
    # NOTE more experiments should be added here under the elif statement with appropriate exp_no and hyperparameters.
    
    else:
        raise ValueError('exp_no %f not defined'%exp_no)
    
    ### Generate (or load) datasets        
    train_data, test_data, train_primary, test_primary, train_secondary, test_secondary = \
                textbox_exp.generate_textbox_data(n=no_data, 
                                                  save=True, 
                                                  save_dir='../data', 
                                                  exp_no=exp_no,
                                                  random_bg=random_bg)
    x_train = train_data.X
    x_test = test_data.X
    y_train = train_data.y
    y_test = test_data.y
    assert(x_test.shape[0] == no_test_per_split * no_split)

    ### Train model   
    print('Training model...')
    train_acc = 0.0
    test_acc = 0.0
    lr_vals = [args.lr]
    if random_bg == 1: # real background
        if exp_no >= 3.5:
            thr = 0.90
        elif exp_no == 1.2:
            thr = 0.90
        else:
            thr = 0.95
    else:
        if exp_no >= 3.5 or exp_no == 1.2:
            thr = 0.95
        else:
            thr = 0.98
    found = False
    prev_test_acc = 0
    
    if not adv_train:
        for ll in lr_vals:
            for i in range(20): 
                if i == 0:
                    retrain = False
                else:
                    retrain = True

                if model_type == 0:
                    if exp_no >= 3.5 or exp_no == 1.2:
                        model_obj = TextBoxCNN_adv(lr=ll, 
                                                model_name=model_name, 
                                                max_epoch=epoch, 
                                                output_dir=CACHE_DIR)
                    else:
                        model_obj = TextBoxCNN(lr=ll, 
                                                model_name=model_name, 
                                                max_epoch=epoch, 
                                                output_dir=CACHE_DIR)
                elif model_type == 1: # vgg
                    model_obj = VGG16_model(lr=ll, model_name=model_name, max_epoch=epoch, output_dir=CACHE_DIR)
                elif model_type == 2: # alexnet
                    model_obj = AlexNet_model(lr=ll, model_name=model_name, max_epoch=epoch, output_dir=CACHE_DIR)
                else:
                    raise ValueError('model_type %d not defined'%model_type)
                
                model_obj.train(x_train, y_train, retrain=retrain, earlystop=True, adversarial=adv_train)
                train_acc = model_obj.test(x_train, y_train) 
                test_acc = model_obj.test(x_test, y_test) 
                model = model_obj.model
                
                if train_acc >= thr and test_acc >= thr: # train until high accuracy
                    found = True
                    break
                else:
                    if test_acc > prev_test_acc:
                        model.save_weights(os.path.join(CACHE_DIR, 'w%0.2f.pt'%exp_no))
                        prev_test_acc = test_acc
                print('-- Retraining due to low performance')
                print('-- Best so far: [%0.4f]'%prev_test_acc)
            if found:
                break
        if not found:
            print('==> Failed to train a model that passes the threshold accuracy [%0.4f]'%thr)
            if prev_test_acc > 1:
                model.load_weights(os.path.join(CACHE_DIR, 'w%0.2f.pt'%exp_no))
                print('==> Loading and proceeding with a model with suboptimal accuracy [%0.4f]'%prev_test_acc)
            else:
                raise ValueError('*** Aborting from training issues ***')
    else:
        # just load from pre-trained data for adversarially robust model (need to run the adv_train.py script first)
        if exp_no >= 3.5 or exp_no == 1.2:
            model_obj = TextBoxCNN_adv(lr=0.001, 
                                    model_name=model_name, 
                                    max_epoch=epoch, 
                                    output_dir=CACHE_DIR)
        else:
            model_obj = TextBoxCNN(lr=0.001, 
                                    model_name=model_name, 
                                    max_epoch=epoch, 
                                    output_dir=CACHE_DIR)
        model_obj.train(x_train, y_train, retrain=False, adversarial=True)
        model = model_obj.model

    ### Run saliency methods
    print('Running Saliency Methods...')
    # How many random images to sample to run the methods on?
    #no_images = int(args.n)
    no_sample_per_block = 20
    result = []
    idx = []
    text = []
    for i in range(no_split):
        ######### HACK: comment this portion to run everything on a single run. 
        ######### Currently, doing so is slower than aborting the process and rerunning due to memory issues.
        exit_after = False
        existing_instance = os.path.exists(os.path.join(CACHE_DIR, 'result_%0.2f_%d.pkl'%(exp_no, i)))
        if not existing_instance:
            exit_after = True
        ################################################################
        print(split_names[i])
        i_start = i * no_test_per_split
        i_end = i_start + no_test_per_split
        _result, _methods, _text, _idx = \
            saliency.run_methods(model, 
                                 x_test[i_start:i_end], 
                                 y_test[i_start:i_end], 
                                 x_train, 
                                 directory=CACHE_DIR, 
                                 no_images=no_sample_per_block, 
                                 exp_no=exp_no, 
                                 load=True, 
                                 split=i,
                                 model_type=model_type)
        ########### HACK: comment this portion to run the whole thing at once. 
        if exit_after:
            assert False, "**\n**\n**Terminating and rerunning from where left off to prevent memory build-up issues.\n**\n**\n**"
        ################################################################
        _idx = _idx + i_start
        result.append(_result)
        idx.append(_idx)
        text.append(_text)
    result = np.array(result)
    result = np.concatenate(result, axis=0)
    result = np.nan_to_num(result) # clip nan values to 0
    idx = np.array(idx)
    idx = np.concatenate(idx, axis=0)
    methods = _methods
    text = np.array(text)
    text = np.concatenate(text, axis=0)
    #print(result.shape, idx.shape)

    result_name = os.path.join(CACHE_DIR, 'result_%0.2f.pkl'%exp_no)
    idx_name = os.path.join(CACHE_DIR, 'idx_%0.2f.pkl'%exp_no)
    methods_name = os.path.join(CACHE_DIR, 'methods_%0.2f.pkl'%exp_no)
    text_name = os.path.join(CACHE_DIR, 'text_%0.2f.pkl'%exp_no)
    pickle.dump(result, open(result_name, 'wb'))
    pickle.dump(idx, open(idx_name, 'wb'))
    pickle.dump(methods, open(methods_name, 'wb'))
    pickle.dump(text, open(text_name, 'wb'))

    print('Computing evaluation metrics...')

    iou_name = os.path.join(CACHE_DIR, 'iou_all_%0.2f.pkl'%exp_no)
    iou_name_single = os.path.join(CACHE_DIR, 'iou_single_all_%0.2f.pkl'%exp_no)
    wiou_name = os.path.join(CACHE_DIR, 'weighted_iou_all_%0.2f.pkl'%exp_no)
    avg_wiou_name = os.path.join(CACHE_DIR, 'avg_weighted_iou_all_%0.2f.pkl'%exp_no)
    
    # load info from the saved bbox list
    print('-- loading bounding box info..')
    bbox_gt_lst = test_primary[idx]
    bbox_avoid_lst = test_secondary[idx]
    
    print('-- processing images from buckets...')
    plain_raw = process_all_bucket(result, bbox_gt_lst, bbox_avoid_lst, no_split, methods, iou_type='plain')
    pickle.dump(plain_raw, open(iou_name, 'wb'))
    plain_single_raw = process_all_bucket(result, bbox_gt_lst, bbox_avoid_lst, no_split, methods, iou_type='plain_single')
    pickle.dump(plain_single_raw, open(iou_name_single, 'wb'))
    weighted_raw = process_all_bucket(result, bbox_gt_lst, bbox_avoid_lst, no_split, methods, iou_type='weighted')
    pickle.dump(weighted_raw, open(wiou_name, 'wb'))
    weighted_raw_avg = process_all_bucket(result, bbox_gt_lst, bbox_avoid_lst, no_split, methods, iou_type='avg-weighted')
    pickle.dump(weighted_raw_avg, open(avg_wiou_name, 'wb'))
    plain_ious = compute_valid_stats(plain_raw)
    plain_single_ious = compute_valid_stats(plain_single_raw)
    weighted_ious = compute_valid_stats(weighted_raw)
    avg_weighted_ious = compute_valid_stats(weighted_raw_avg)
    
    # AFL
    m_piou, s_piou = weighted_ious[0]
    m_siou, s_siou = weighted_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_weighted_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou, s_piou, m_siou, s_siou])
    
    # IOU
    m_piou_plain_single, s_piou_plain_single = plain_single_ious[0]
    m_siou_plain_single, s_siou_plain_single = plain_single_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_plain_single_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou_plain_single, s_piou_plain_single, m_siou_plain_single, s_siou_plain_single])
    
    # IOU multi-thresholded
    m_piou_plain, s_piou_plain = plain_ious[0]
    m_siou_plain, s_siou_plain = plain_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_plain_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou_plain, s_piou_plain, m_siou_plain, s_siou_plain])

    # MAFL
    m_avg_piou, s_avg_piou = avg_weighted_ious[0]
    m_avg_siou, s_avg_siou = avg_weighted_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_avg_weighted_%0.2f.npy'%exp_no), 'wb'), \
        [m_avg_piou, s_avg_piou, m_avg_siou, s_avg_siou])
    
    print('==> All metrics computed for all buckets.')

    print('Plotting...')
    #Plot metrics overall
    print('-- PIOU - multithresholding')
    plot_iou(methods, m_piou_plain, s_piou_plain, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou'%exp_no))
    print('-- SIOU - multithresholding')
    plot_iou(methods, m_siou_plain, s_siou_plain, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid'%exp_no))

    print('-- PIOU - singlethresholding')
    plot_iou(methods, m_piou_plain_single, s_piou_plain_single, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_single'%exp_no))
    print('-- SIOU - singlethresholding')
    plot_iou(methods, m_siou_plain_single, s_siou_plain_single, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_single'%exp_no))

    print('-- PAFL')
    plot_iou(methods, m_piou, s_piou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_weighted'%exp_no))
    print('-- SAFL')
    plot_iou(methods, m_siou, s_siou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_weighted'%exp_no))

    print('-- PMAFL')
    plot_iou(methods, m_avg_piou, s_avg_piou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avg_weighted'%exp_no))
    print('-- SMAFL')
    plot_iou(methods, m_avg_siou, s_avg_siou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_avg_weighted'%exp_no))

    # Plot across different groups with different feature
    print('-- Per Bucket Information')
    eval_per_bucket(weighted_raw, 
                    no_split, 
                    methods, 
                    split_names, 
                    exp_no,
                    plot_dir=PLOT_DIR,
                    cache_dir=CACHE_DIR,
                    iou_type='weighted')
    eval_per_bucket(plain_raw, 
                    no_split, 
                    methods, 
                    split_names, 
                    exp_no,
                    plot_dir=PLOT_DIR,
                    cache_dir=CACHE_DIR,
                    iou_type='plain')
    eval_per_bucket(plain_single_raw, 
                    no_split, 
                    methods, 
                    split_names, 
                    exp_no,
                    plot_dir=PLOT_DIR,
                    cache_dir=CACHE_DIR,
                    iou_type='plain_single')
    eval_per_bucket(weighted_raw_avg, 
                    no_split, 
                    methods, 
                    split_names, 
                    exp_no,
                    plot_dir=PLOT_DIR,
                    cache_dir=CACHE_DIR,
                    iou_type='avg-weighted')
    
    print('============= exp [%0.2f] Done =============='%(exp_no))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=float, help='experiment number. 1-FR, 2-NR, 3-CR')
    parser.add_argument('--n', type=float, default=100, help='number of images to run experiments on per group')
    parser.add_argument('--ep', type=int, default=10, help='max epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--bg', type=int, default=0, help='set this to 1 for natural background, 0 for zero background')
    parser.add_argument('--model_type', type=int, default=0, help='set this to 0 for simple CNN, 1 for vgg16, 2 for AlexNet for the base model')
    parser.add_argument('--adv', type=int, default=0, help='set this to 0 for standard model training, 1 adversarial training')
    args = parser.parse_args()
    print(args)
    main(args)

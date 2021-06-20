import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../')
import smerf
from smerf.models import *
from smerf.textcolor_data import *
from smerf.eval import *
import smerf.explanations as saliency

CACHE_DIR = '../outputs/cache'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
    
PLOT_DIR = '../outputs/plots'
if not os.path.exists(PLOT_DIR):
    os.mkdir(PLOT_DIR)

import argparse

def main(args):
    exp_no = args.exp
    model_name = 'w%0.2f.pt'%exp_no
    print('EXP_NO = %f'%exp_no)
    lr = args.lr
    epoch = args.ep

    ### Below defines the hyperparameters for each experiment setups. 
    ### For new experiments, these should be stated explicitly. 
    if exp_no == 1.11: ## Simple-FR: moving patch (white)
        import smerf.simple_fr as textcolor_exp
        no_data = 2000
        no_test_per_split = 500
        no_split = 12
        other_methods = []
        enforce = None
        ## define split names
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

    elif exp_no == 2.11: ## Simple-NR: patch and switch (moving) (white text, black background)
        #import smerf.textcolor_nr_complex2 as textcolor_exp
        import smerf.simple_nr as textcolor_exp
        no_data = 5000
        no_test_per_split = 500
        no_split = 8
        other_methods = []
        enforce = None
        ## define split names
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

    elif exp_no == 1.2: ## Complex-FR: complex (two ground-truth objects)
        import smerf.complex_fr as textcolor_exp
        no_data = 2000
        no_test_per_split = 500
        no_split = 12
        other_methods = []
        enforce = None
        ## define split names
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
        import smerf.complex_cr1 as textcolor_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        other_methods = []
        ## define split names
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
        import smerf.complex_cr2 as textcolor_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        other_methods = []
        ## define split names
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
        import smerf.complex_cr3 as textcolor_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        other_methods = []
        ## define split names
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
        import smerf.complex_cr4 as textcolor_exp
        no_data = 15000
        no_test_per_split = 400
        no_split = 10
        other_methods = []
        ## define split names
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

    # NOTE more experiments should be added below
    else:
        raise ValueError('exp_no %f not defined'%exp_no)

    ### Create datasets
    if exp_no >= 3.5:
        train_data, test_data, train_coord, test_coord, train_avoid, test_avoid, train_enforce, test_enforce = \
            textcolor_exp.spurious_textcolor_data(n=no_data, 
                                                  save=True, 
                                                  save_dir='../data', 
                                                  exp_no=exp_no) 
    else:
        train_data, test_data, train_coord, test_coord, train_avoid, test_avoid, train_enforce, test_enforce = \
            textcolor_exp.spurious_textcolor_data(n=no_data, 
                                                  save=True, 
                                                  save_dir='../data', 
                                                  exp_no=exp_no)
    
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
    found = False
    for ll in lr_vals:
        for i in range(50):
            if i == 0:
                retrain = False
            else:
                retrain = True
            if exp_no >= 3.5 or exp_no == 1.2:
                model_obj = TextColorCNN_adv(lr=ll, 
                                        model_name=model_name, 
                                        max_epoch=epoch, 
                                        output_dir=CACHE_DIR)
                if exp_no >= 3.5:
                    thr = 0.95
                else:
                    thr = 0.99
            else:
                model_obj = TextColorCNN(lr=ll, 
                                        model_name=model_name, 
                                        max_epoch=epoch, 
                                        output_dir=CACHE_DIR)
                thr = 0.995
            model_obj.train(x_train, y_train, retrain=retrain, earlystop=True)
            train_acc = model_obj.test(x_train, y_train)
            test_acc = model_obj.test(x_test, y_test)
            model = model_obj.model
            if train_acc >= thr and test_acc >= thr: # train until high accuracy
                found = True
                break
        if found:
            break
    if not found:
        raise ValueError('failed to train a model')

    ### Run saliency methods
    print('Running Saliency Methods...')
    # How many random images to sample to run the methods on?
    #no_images = int(args.n)
    no_sample_per_block = 20
    result = []
    idx = []
    text = []
    for i in range(no_split):
        ######### HACK: uncomment to run the whole thing at once
        exit_after = False
        existing_instance = os.path.exists(os.path.join(CACHE_DIR, 'result_%0.2f_%d.pkl'%(exp_no, i)))
        if not existing_instance:
            exit_after = True
        #########
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
                                 split=i)
        ########### HACK: uncomment this to run the whole thing at once (slow due to memory issues)
        if exit_after:
            assert(False)
        ###########
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
    print(result.shape, idx.shape)

    result_name = os.path.join(CACHE_DIR, 'result_%0.2f.pkl'%exp_no)
    idx_name = os.path.join(CACHE_DIR, 'idx_%0.2f.pkl'%exp_no)
    methods_name = os.path.join(CACHE_DIR, 'methods_%0.2f.pkl'%exp_no)
    text_name = os.path.join(CACHE_DIR, 'text_%0.2f.pkl'%exp_no)
    pickle.dump(result, open(result_name, 'wb'))
    pickle.dump(idx, open(idx_name, 'wb'))
    pickle.dump(methods, open(methods_name, 'wb'))
    pickle.dump(text, open(text_name, 'wb'))

    print('computing evaluation metrics...')

    iou_name = os.path.join(CACHE_DIR, 'iou_all_%0.2f.pkl'%exp_no)
    iou_name_single = os.path.join(CACHE_DIR, 'iou_single_all_%0.2f.pkl'%exp_no)
    wiou_name = os.path.join(CACHE_DIR, 'weighted_iou_all_%0.2f.pkl'%exp_no)
    avg_wiou_name = os.path.join(CACHE_DIR, 'avg_weighted_iou_all_%0.2f.pkl'%exp_no)

    if exp_no >= 3.5:
        bbox_gt_lst, bbox_avoid_lst = setup_bboxes(test_coord, test_avoid, test_enforce, idx, gt_flag=[1,0,1])
    elif exp_no == 1.2: # complex_fr: multiple GT
        bbox_gt_lst, bbox_avoid_lst = setup_bboxes(test_coord, test_avoid, test_enforce, idx, gt_flag=[1,0,1])
    else:
        bbox_gt_lst, bbox_avoid_lst = setup_bboxes(test_coord, test_avoid, test_enforce, idx, gt_flag=[1,0,0])
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

    m_piou, s_piou = weighted_ious[0]
    m_siou, s_siou = weighted_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_weighted_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou, s_piou, m_siou, s_siou])

    m_piou_plain_single, s_piou_plain_single = plain_single_ious[0]
    m_siou_plain_single, s_siou_plain_single = plain_single_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_plain_single_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou_plain_single, s_piou_plain_single, m_siou_plain_single, s_siou_plain_single])

    m_piou_plain, s_piou_plain = plain_ious[0]
    m_siou_plain, s_siou_plain = plain_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_plain_%0.2f.npy'%exp_no), 'wb'), \
        [m_piou_plain, s_piou_plain, m_siou_plain, s_siou_plain])

    m_avg_piou, s_avg_piou = avg_weighted_ious[0]
    m_avg_siou, s_avg_siou = avg_weighted_ious[1]
    np.save(open(os.path.join(CACHE_DIR, 'metrics_avg_weighted_%0.2f.npy'%exp_no), 'wb'), \
        [m_avg_piou, s_avg_piou, m_avg_siou, s_avg_siou])

    print('plotting...')
    #Plot IOU overall
    print(' - IOU')
    plot_iou(methods, m_piou_plain, s_piou_plain, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou'%exp_no))
    print(' - IOU-Avoid')
    plot_iou(methods, m_siou_plain, s_siou_plain, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid'%exp_no))

    print(' - IOU')
    plot_iou(methods, m_piou_plain_single, s_piou_plain_single, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_single'%exp_no))
    print(' - IOU-Avoid')
    plot_iou(methods, m_siou_plain_single, s_siou_plain_single, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_single'%exp_no))

    print(' - wIOU')
    plot_iou(methods, m_piou, s_piou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_weighted'%exp_no))
    print(' - wIOU-Avoid')
    plot_iou(methods, m_siou, s_siou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_weighted'%exp_no))

    print(' - avg wIOU')
    plot_iou(methods, m_avg_piou, s_avg_piou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avg_weighted'%exp_no))
    print(' - avg wIOU-Avoid')
    plot_iou(methods, m_avg_siou, s_avg_siou, fname=os.path.join(PLOT_DIR, 'exp%0.2f_iou_avoid_avg_weighted'%exp_no))

    # Plot samples per group
    print(' - samples')
    for i in range(no_split):
        print(split_names[i])
        i_start = i * no_sample_per_block
        i_end = i_start + 10
        fname = os.path.join(PLOT_DIR, 'samples_%0.2f_%d'%(exp_no, i))
        smerf.eval.visualize_examples(result[i_start:i_end], 
                                      methods, 
                                      text[i_start:i_end], 
                                      idx[i_start:i_end], 
                                      fname=fname)

    # Plot across different groups with different feature
    print(' - different groups')
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
    args = parser.parse_args()
    print(args)
    main(args)

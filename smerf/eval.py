import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import imp
import os, pickle
from matplotlib.backends.backend_pdf import PdfPages   
textcolorutils = imp.load_source('textcolor_utils', '../smerf/textcolor_utils.py')

### TextBox Data Evaluations ###

def get_binary_masks(bbox, shape):
    mask = np.zeros(shape)
    mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1.0
    return mask

# Define a method to compute AFL
def get_afl(sal, bbox_lst, methods, method_idx):
    if methods[method_idx][2] == textcolorutils.heatmap:
        pp = 1 - np.mean(sal, axis=2)
    else:
        pp = np.max(sal, axis=2)
    pp = cv2.blur(pp, (7,7)) # blur
    pp = (pp - np.min(pp)) / (np.max(pp) - np.min(pp)) # normalize values [0,1]
    pp = pp / pp.sum()
    
    no_objs = len(bbox_lst)
    score_vals = []
    if no_objs == 0: # if there is no object, return invalid value
        return -1
    for i in range(no_objs):
        bbox = bbox_lst[i]
        mask = get_binary_masks(bbox, pp.shape)
        inters = pp * mask
        score = inters[bbox[0]:bbox[1], bbox[2]:bbox[3]].sum()
        score_vals.append(score)
    return sum(score_vals)

# Define a method to compute MAFL
def get_mafl(sal, bbox_lst, methods, method_idx):
    if methods[method_idx][2] == textcolorutils.heatmap:
        pp = 1 - np.mean(sal, axis=2)
    else:
        pp = np.max(sal, axis=2)
    pp = cv2.blur(pp, (7,7)) # blur
    pp = (pp - np.min(pp)) / (np.max(pp) - np.min(pp)) # normalize values [0,1]
    pp = pp / pp.sum()
    
    no_objs = len(bbox_lst)
    score_vals = []
    if no_objs == 0: # if there is no object, return invalid value
        return -1
    for i in range(no_objs):
        bbox = bbox_lst[i]
        mask = get_binary_masks(bbox, pp.shape)
        inters = pp * mask
        score = inters[bbox[0]:bbox[1], bbox[2]:bbox[3]].flatten()
        score_vals.append(score)
    return np.mean(np.concatenate(score_vals))

# Define a method to compute plain IOU
def get_plain_iou(sal, bbox_lst, methods, method_idx, pixel_no):
    no_objs = len(bbox_lst)
    score_vals = []
    if no_objs == 0: # if there is no object, return invalid value
        return -1
    if methods[method_idx][2] == textcolorutils.heatmap:
        pp = 1 - np.mean(sal, axis=2)
    else:
        pp = np.max(sal, axis=2)
    pp = cv2.blur(pp, (7,7)) # blur
    vals, bins = np.histogram(pp.flatten()) # historgram of pixels
    thr = get_thrs(vals, bins, pixel_no=pixel_no) # find threshold to be the value that contains around some pixels
    pp = np.array(pp >= thr) * 1.0 # mask the image with the threshold 
    if np.isnan(sal).all():
        pp = np.zeros(64,64)
    for i in range(no_objs):
        bbox = bbox_lst[i]
        mask = get_binary_masks(bbox, pp.shape)
        score = np.where(mask + pp == 2)[0].shape[0] / (np.where(mask + pp != 0)[0].shape[0])
        score_vals.append(score)
    return np.mean(score_vals)

# Compute metric values for each bucket. 
def process_bucket(bucket_result, bbox_gt_lst, bbox_avoid_lst, methods, iou_type='weighted'):
    no_images, no_methods, _, _, _ = bucket_result.shape
    output = np.zeros((no_images, no_methods-1, 2))
    for i in range(no_images):
        for j in range(1,no_methods):
            if iou_type == 'weighted': # AFL
                piou = get_afl(bucket_result[i,j], bbox_gt_lst[i], methods, j)
                siou = get_afl(bucket_result[i,j], bbox_avoid_lst[i], methods, j)

            elif iou_type == 'avg-weighted': # MAFL
                piou = get_mafl(bucket_result[i,j], bbox_gt_lst[i], methods, j)
                siou = get_mafl(bucket_result[i,j], bbox_avoid_lst[i], methods, j)
                
            elif iou_type == 'plain': # plain-IOU
                pixel_no = get_pixel_no(bbox_gt_lst[i])
                piou = get_plain_iou(bucket_result[i,j], bbox_gt_lst[i], methods, j, pixel_no)
                pixel_no = get_pixel_no(bbox_avoid_lst[i])
                siou = get_plain_iou(bucket_result[i,j], bbox_avoid_lst[i], methods, j, pixel_no)

            elif iou_type == 'plain_single': # single threshold based on the primary bbox
                pixel_no = get_pixel_no(bbox_gt_lst[i])
                piou = get_plain_iou(bucket_result[i,j], bbox_gt_lst[i], methods, j, pixel_no)
                siou = get_plain_iou(bucket_result[i,j], bbox_avoid_lst[i], methods, j, pixel_no)

            else:
                raise ValueError('%s metric type not supported'%iou_type)
    
            output[i,j-1,0] = piou
            output[i,j-1,1] = siou
    return output

def get_pixel_no(bbox_lst):
    no = 0
    for i in range(len(bbox_lst)):
        no += (bbox_lst[i][1] - bbox_lst[i][0]) * (bbox_lst[i][3] - bbox_lst[i][2])
    return no 

def process_all_bucket(result, bbox_gt_lst, bbox_avoid_lst, bucket_no, methods, iou_type='weighted'):
    no_images, no_methods, _, _, _ = result.shape
    samples_per = int(no_images / bucket_no)
    output = np.zeros((no_images, no_methods-1, 2))
    for b in range(bucket_no):
        start_idx = samples_per * b
        end_idx = start_idx + samples_per
        bucket_result = result[start_idx:end_idx]
        bbox_gt_bucket = bbox_gt_lst[start_idx:end_idx]
        bbox_avoid_bucket = bbox_avoid_lst[start_idx:end_idx]
        bucket_ious = process_bucket(bucket_result, bbox_gt_bucket, bbox_avoid_bucket, methods, iou_type=iou_type)
        output[start_idx:end_idx] = bucket_ious
    return output

def compute_valid_stats(arr):
    # ignore invalid ones and compute avg and std
    no_images, no_methods, no_values = arr.shape
    out = []
    for i in range(no_values):
        tt = arr[:,:,i]
        tt = tt[~(tt == -1).any(axis=1)]
        out.append((np.mean(tt, axis=0), np.std(tt, axis=0)))
    return out

# get the bbox_lst set up (for one gt)
def setup_bboxes(test_coords, test_avoid, test_avoid2, idx, gt_flag=[1,0,0]):
    inputs = [test_coords[idx], test_avoid[idx], test_avoid2[idx]]
    include = []
    exclude = []
    for i in range(len(gt_flag)):
        if gt_flag[i] == 1:
            include.append(inputs[i])
        else:
            exclude.append(inputs[i])

    bbox_gt_lst = []
    bbox_avoid_lst = []
    for i in range(idx.shape[0]):
        tmp = []
        for inc in include:
            if inc[i][0] != None:
                tmp.append(inc[i])
        bbox_gt_lst.append(tmp)
        tmp = []
        for exc in exclude:
            if exc[i][0] != None:
                tmp.append(exc[i])
        bbox_avoid_lst.append(tmp)

    return bbox_gt_lst, bbox_avoid_lst

# post-processing as a separate function
def post_process_saliency(sal, img_idx, method_idx, methods, pixel_nos, enforce=None):
    """
    Post-process the saliency output for computing metrics.

    :param sal: (H, W, C) array, saliency output for a single image, using a single method
    :param img_idx: index of the current image
    :param method_idx: index of the current method
    :param pixel_nos: number of pixels to include for all N images
    :param enforce: [(rmin, rmax, cmin, cmax)], indicating list of regions where the saliency value should be clipped at zero

    :return: post-processed saliency output, binarized (H, W) array, for metric computation
    """
    if methods[method_idx][2] == textcolorutils.heatmap:
        pp = 1 - np.mean(sal, axis=2)
    else:
        pp = np.max(sal, axis=2)
        #pp = sal.sum(-1)
    if enforce is not None:
    # for CR case, need to consider exceptional cases (where the switch region should be manually reset to zero )
        for ee in enforce:
            if None in ee:
                pass
            else:
                pp[ee[0]:ee[1], ee[2]:ee[3]] = 0.0
    pp = cv2.blur(pp, (7,7)) # blur
    if np.max(pp) - np.min(pp) != 0:
        pp = (pp - np.min(pp)) / (np.max(pp) - np.min(pp)) # normalize values [0,1]
    vals, bins = np.histogram(pp.flatten()) # historgram of pixels
    pixel_no = pixel_nos[img_idx] # different pixel no for each image
    thr = get_thrs(vals, bins, pixel_no=pixel_no) # find threshold to be the value that contains around some pixels
    pp = np.array(pp >= thr) * 1.0 # mask the image with the threshold 
    return pp

def compute_ssim(sal_thr, bbox):
    """
    Compute SSIM metric

    :param sal_thr: post-processed saliency output, binarized (H,W) array
    :param bbox: (rmin, rmax, cmin, cmax), ground-truth bounding box

    :return: SSIM metric value between input saliency and target saliency
    """
    # using the bounding box, create a 0-1 masked saliency
    target = np.zeros(sal_thr.shape)
    target[bbox[0]:bbox[1], bbox[2]:bbox[3]] = 1.0
    return ssim(sal_thr, target, multichannel=False)

def get_bbox(img):
    """
    Compute the bounding box coordinates of the given thresholded saliency input.
    Bounding box is determined as the smallest rectangle that contains all
    relevant pixels in the input.

    :param img: post-processed saliency output, binarized (H,W) array

    :return: (rmin, rmax, cmin, cmax) bounding box detected
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def compute_iou(bbox1, bbox2):
    """
    Compute IOU metric value (as well as precision and recall) between two bounding boxes.

    param bbox1: (rmin, rmax, cmin, cmax) detected bounding box 
    param bbox2: (rmin, rmax, cmin, cmax) ground-truth bounding box

    return: (IOU metric, precision, recall) tuple of results
    """
    # intersection rectangle coordinates
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[3], bbox2[3])
    y2 = min(bbox1[1], bbox2[1])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (bbox1[1] - bbox1[0] + 1) * (bbox1[3] - bbox1[2] + 1) # detection
    area2 = (bbox2[1] - bbox2[0] + 1) * (bbox2[3] - bbox2[2] + 1) # real

    # Precision: intersection / detection_bounding_box
    # Recall: intersection / real_object_boudnding_box
    prec = intersection / area1
    rec = intersection / area2

    return intersection / (area1 + area2 - intersection), prec, rec

def get_thrs(vals, bins, pixel_no=100):
    """
    Compute appropriate threshold value to use for post-processing.
    The threshold is computed so that it should include the pixel number specified. 

    :param vals: values obtained from histogram of pixel values
    :param bins: bins obtained from histogram of pixel values
    :param pixel_no: number of pixels to include when thresholded

    :return: threshold value to use for post-processing
    """
    c = 0
    for i in range(len(vals)-1, -1, -1):
        c += vals[i]
        if c >= pixel_no:
            thr = bins[i]
            break
    return thr

### Plotting functions ####

# Visualize examples from the result array
def visualize_examples(result, methods, text, idx, fname='result'):
    """
    Visualize examples from the result array.

    :param result: (N, M, H, W, C) array of saliency outputs for each image and each method
    :param methods: list of methods
    :param text: list of texts to be used for labels
    :param idx: list of indices from the test data that were selected for plotting
    :param fname: (optional) file name to save the plot to
    """
    no_methods = result.shape[1]
    no_images = result.shape[0]
    assert(len(idx) == no_images)
    f, axs = plt.subplots(no_images, no_methods, figsize=(2*no_methods+1, 2*no_images))
    for i in range(no_images):
        for j in range(no_methods):
            axs[i,j].imshow(result[i, j])
            axs[i,j].set_xticks(())
            axs[i,j].set_yticks(())
            if j == 0:
                axs[i,j].set_ylabel('id%d\nlabel=%s\npred=%s'%(idx[i], text[i][0], text[i][3]))
            if i == 0:
                axs[i,j].set_title(methods[j][3], rotation=45)
    plt.tight_layout()
    plt.savefig(fname+'.pdf', dpi=200)

# Plot IOU metric values 
def plot_iou(methods, m, s, fname='exp0_iou', title=None, axs=None, save=True):
    if axs is None:
        f, axs = plt.subplots(1,1, figsize=(4,3))
    names = [x[3] for x in methods][1:]
    axs.bar(range(len(m)), m, yerr=s, tick_label=names)
    axs.set_ylim([0.0,1.0])
    axs.set_xticklabels(names, rotation=45, ha='right')
    if title is None:
        axs.set_title('IOU Values')
    else:
        axs.set_title(title)
    if save:
        plt.tight_layout()
        plt.savefig(fname+'.pdf', dpi=200)
    return axs

# plot all metrics together
def plot_all(methods, m_iou, s_iou, m_prec, s_prec, m_rec, s_rec, m_ssim, s_ssim, fname='plot_all'):
    names = [x[3] for x in methods][1:]
    f, axs = plt.subplots(1,4, figsize=(4*4, 5))
    axs[0].barh(range(len(m_iou)), m_iou, xerr=s_iou, tick_label=names)
    axs[0].set_xlim([0.0, 1.0])
    axs[0].set_title('IOU')
    axs[1].barh(range(len(m_prec)), m_prec, xerr=s_prec)
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_yticks(())
    axs[1].set_title('Precision')
    axs[2].barh(range(len(m_rec)), m_rec, xerr=s_rec)
    axs[2].set_xlim([0.0, 1.0])
    axs[2].set_yticks(())
    axs[2].set_title('Recall')
    axs[3].barh(range(len(m_ssim)), m_ssim, xerr=s_ssim)
    axs[3].set_xlim([0.0, 1.0])
    axs[3].set_title('SSIM')
    axs[3].set_yticks(())
    plt.tight_layout()
    plt.savefig(fname+'.pdf', dpi=200)
    return f, axs

# Plot results per bucket
def eval_per_bucket(vals, bucket_no, methods, split_names, exp_no, plot_dir='', cache_dir='', iou_type='weighted'):
    no_images, no_methods, no_vals = vals.shape
    samples_per = int(no_images / bucket_no)
    plot_fname = os.path.join(plot_dir, 'iou_%0.2f_per_bucket_%s.pdf'%(exp_no, iou_type))
    num_col = 6
    num_row = int(np.ceil(bucket_no / 6))
    f1, axs1 = plt.subplots(num_row, num_col, figsize=(4 * num_col, 4 * num_row))
    f2, axs2 = plt.subplots(num_row, num_col, figsize=(4 * num_col, 4 * num_row))
    
    for b in range(bucket_no):
        print(' plotting bucket: %s'%(split_names[b]))
        start_idx = samples_per * b
        end_idx = start_idx + samples_per
        bucket_stats = compute_valid_stats(vals[start_idx:end_idx])
        m_iou, s_iou = bucket_stats[0]
        m_siou, s_siou = bucket_stats[1]
        np.save(open(os.path.join(cache_dir, 'metrics_%s_%0.2f_%d.npy'%(iou_type, exp_no, b)), 'wb'), \
                        [m_iou, s_iou, m_siou, s_siou])
        
        if b < num_col:
            i, j = 0, b
        else:
            i, j = int(b/num_col), int(b % num_col)
            
        if num_row == 1:
            if not np.isnan(m_iou).any():
                _ = plot_iou(methods, m_iou, s_iou, fname=None, save=False, axs=axs1[j])
            axs1[j].set_title(split_names[b], fontsize=15)
            if not np.isnan(m_siou).any():
                _ = plot_iou(methods, m_siou, s_siou, fname=None, save=False, axs=axs2[j])
            axs2[j].set_title(split_names[b], fontsize=15)
        else:
            if not np.isnan(m_iou).any():
                _ = plot_iou(methods, m_iou, s_iou, fname=None, save=False, axs=axs1[i,j])
            axs1[i,j].set_title(split_names[b], fontsize=15)
            if not np.isnan(m_siou).any():
                _ = plot_iou(methods, m_siou, s_siou, fname=None, save=False, axs=axs2[i,j])
            axs2[i,j].set_title(split_names[b], fontsize=15)
    plt.tight_layout()
    pp = PdfPages(plot_fname)
    pp.savefig(f1)
    pp.savefig(f2)
    pp.close()
    plt.close()
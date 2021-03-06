from .textbox_data import *
from .eval import *
import matplotlib.pyplot as plt

### EXP1.2 Full reliance (complex): two ground-truths: patch and switch
def generate_textbox_data(n=10000, save=True, save_dir='data', exp_no=1.2, random_bg=False):
    def create_data(patch, text, switch, n=n):
        print('Creating %d images with SWITCH=%d / PATCH=%d / TEXT=%d...'%(n, switch, patch, text))

        def create_single_img(switch, patch, text):
            feature = np.zeros((6))
            feature[0] = text # -1 = no text, 0 = A, 1 = B
            feature[1] = np.random.uniform() #x
            feature[2] = np.random.uniform() #y
            feature[3] = 3 # character color white
            if random_bg:
                feature[4] = -4 # natural background
            else:
                feature[4] = -2 # black background
            feature[5] = patch
            if patch:
                # random patch location
                while True:
                    x_start = np.random.random_integers(0, 53)
                    y_start = np.random.random_integers(0, 53)
                    # prevent overlap of patch with text
                    if (int(36*feature[1]) - x_start)**2 + (int(36*feature[2]) - y_start)**2 > 300:
                        break
                im, dim, sw = vec2im(feature, x_start=x_start, y_start=y_start, switch=switch, s_color=(255,255,255), p_color=(255,255,255))
                x_end = x_start+10
                y_end = y_start+10
            else:
                im, dim, sw = vec2im(feature, switch=switch, s_color=(255, 255, 255), p_color=(255,255,255))
                x_start = None
                y_start = None
                x_end = None
                y_end = None
            # specify the label according to the model reasoning desired
            if patch == 1 and switch == 2:
                label = 1
                bbox1 = (y_start, y_end, x_start, x_end)
                bbox2 = dim
                bbox3 = sw
            else:
                label = 0
                bbox1 = (None, None, None, None)
                bbox2 = dim
                bbox3 = (None, None, None, None)

            return np.float32(im / 255), label, bbox1, bbox2, bbox3
            #return np.float32(im / 255), label, (y_start, y_end, x_start, x_end), dim, sw

        X = []
        y = []
        bboxes = []
        bbox_avoid = []
        bbox_avoid2 = []
        for _ in range(n):
            xx, yy, meta, bbavoid, bbavoid2 = create_single_img(switch, patch, text)
            X.append(xx)
            y.append(yy)
            bboxes.append(meta)
            bbox_avoid.append(bbavoid)
            bbox_avoid2.append(bbavoid2)
        X = np.array(X)
        y = np.array(y)
        bboxes = np.array(bboxes)
        bbox_avoid = np.array(bbox_avoid)
        bbox_avoid2 = np.array(bbox_avoid2)
        return X, y, bboxes, bbox_avoid, bbox_avoid2

    def create_dataset(n=n, train=False):
        # Go through all possible cases
        data = []
        labels = []
        bboxes = []
        bboxes_avoid = []
        bboxes_avoid2 = []
        patch = [0,1]
        text = [-1,0,1]
        switch = [0,2]

        for pa in patch:
            for la in text:
                for sw in switch:
                    if pa == 1 and sw == 2 and train:
                        X, y, bbox, bbox_avoid, bbox_avoid2 = create_data(pa, la, sw, n=3*n)
                    else:
                        X, y, bbox, bbox_avoid, bbox_avoid2 = create_data(pa, la, sw, n=n)
                    data.append(X)
                    labels.append(y)
                    bboxes.append(bbox)
                    bboxes_avoid.append(bbox_avoid)
                    bboxes_avoid2.append(bbox_avoid2)
        X = np.concatenate(data, axis=0)
        y = np.concatenate(labels, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        bboxes_avoid = np.concatenate(bboxes_avoid, axis=0)
        bboxes_avoid2 = np.concatenate(bboxes_avoid2, axis=0)
        return TextBoxDataset(X, y), bboxes, bboxes_avoid, bboxes_avoid2

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    fname = os.path.join(save_dir, 'textbox_%0.2f.npz'%exp_no) 
    if os.path.exists(fname): # if the file exists, load
        print("Loading from cached data")
        train_data, test_data, train_primary, train_secondary, test_primary, test_secondary = \
            load_data(exp_no, save_dir)
    else: # otherwise create the new dataset from scratch
        print('Generating data from scratch')
        train_data, train_coord, train_avoid, train_avoid2 = create_dataset(n=n, train=True)
        test_data, test_coord, test_avoid, test_avoid2 = create_dataset(n=500)
        train_data, test_data, train_primary, train_secondary, test_primary, test_secondary = \
            save_data(exp_no, save_dir, train_data, test_data, train_coord, train_avoid, \
                      train_avoid2, test_coord, test_avoid, test_avoid2, save=save) 
        
    return train_data, test_data, train_primary, test_primary, train_secondary, test_secondary
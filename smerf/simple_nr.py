from .textbox_data import *
from .eval import *
import matplotlib.pyplot as plt

### EXP2.11 No Reliance (simple)
def generate_textbox_data(n=3000, save=True, save_dir='data', exp_no=2.11):
    # sets of features to be corrleated: switch feature (binary), patch feature (binary), text (binary)
    def create_data(switch, patch, text, n=n):
        # create images under specific conitions

        print('Creating %d images with PATCH=%d / SWITCH=%d / TEXT=%d...'%(n, patch, switch, text))

        def create_single_img(switch, patch, text):
            feature = np.zeros((6))
            feature[0] = text # -1 = no text, 0 = A, 1 = B
            feature[1] = np.random.uniform() #x
            feature[2] = np.random.uniform() #y
            feature[3] = 3 # character color white
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
            label = text
            return np.float32(im / 255), label, dim, (y_start, y_end, x_start, x_end), sw

        X = []
        y = []
        bboxes = []
        bboxes_avoid = []
        bboxes_avoid2 = []
        for _ in range(n):
            xx, yy, dd, bbox_avoid, bbox_avoid2  = create_single_img(switch, patch, text)
            X.append(xx)
            y.append(yy)
            bboxes.append(dd)
            bboxes_avoid.append(bbox_avoid)
            bboxes_avoid2.append(bbox_avoid2)
        X = np.array(X)
        y = np.array(y)
        bboxes = np.array(bboxes)
        bboxes_avoid = np.array(bboxes_avoid)
        bboxes_avoid2 = np.array(bboxes_avoid2)
        return X, y, bboxes, bboxes_avoid, bboxes_avoid2

    def create_dataset(n=n):
        # Go through all possible cases
        data = []
        labels = []
        bboxes = []
        bboxes_avoid = []
        bboxes_avoid2 = []
        switch = [0,2]
        patch = [0,1]
        text = [0,1]
        for sw in switch:
            for pa in patch:
                for la in text:
                    X, y, bbox, bbox_avoid, bbox_avoid2 = create_data(sw, pa, la, n=n)
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
        train_data, train_coord, train_avoid, train_avoid2 = create_dataset(n=n)
        test_data, test_coord, test_avoid, test_avoid2 = create_dataset(n=500)
        train_data, test_data, train_primary, train_secondary, test_primary, test_secondary = \
            save_data(exp_no, save_dir, train_data, test_data, train_coord, train_avoid, \
                      train_avoid2, test_coord, test_avoid, test_avoid2, save=save) 
        
    return train_data, test_data, train_primary, test_primary, train_secondary, test_secondary
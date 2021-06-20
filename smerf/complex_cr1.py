from .textcolor_data import *
from .eval import *
import matplotlib.pyplot as plt

### EXP 3.71 Conditional Reliance 1
def spurious_textcolor_data(n=10000, save=True, save_dir='data', exp_no=3.71):
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
                im, dim, sw = vec2im(feature, switch=switch, s_color=(255,255,255), p_color=(255,255,255))
                x_start = None
                y_start = None
                x_end = None
                y_end = None

            # specify the label according to the model reasoning desired
            if switch==2:
                if patch:
                    label = 0
                else:
                    label = 1
            else:
                label = text

            # Generate ground-truth bboxes for saliency evaluation
            if switch == 0:
                bbox = dim
                bbox_avoid = (y_start, y_end, x_start, x_end)
                bbox_enforce = (None, None, None, None)
            else:
                bbox = (y_start, y_end, x_start, x_end) # patch location
                bbox_avoid = dim
                bbox_enforce = sw
            return np.float32(im / 255), label, bbox, bbox_avoid, bbox_enforce

        X = []
        y = []
        bbox = []
        bboxes_avoid = []
        bboxes_enforce = []
        for _ in range(n):
            xx, yy, bb, bbox_avoid, bbox_enforce = create_single_img(switch, patch, text)
            X.append(xx)
            y.append(yy)
            bbox.append(bb)
            bboxes_avoid.append(bbox_avoid)
            bboxes_enforce.append(bbox_enforce)
        X = np.array(X)
        y = np.array(y)
        bbox = np.array(bbox)
        bboxes_avoid = np.array(bboxes_avoid)
        bboxes_enforce = np.array(bboxes_enforce)
        return X, y, bbox, bboxes_avoid, bboxes_enforce

    def create_dataset(n=n):
        # Go through all possible cases
        data = []
        labels = []
        bboxes = []
        bboxes_avoid = []
        bboxes_enforce = []
        switch = [0,2]
        patch = [0,1]
        text = [-1,0,1]
        for sw in switch:
            for pa in patch:
                for la in text:
                    if la == -1 and sw == 0: # undecided regions
                        continue
                    X, y, bbox, bbox_avoid, bbox_enforce = create_data(sw, pa, la, n=n)
                    data.append(X)
                    labels.append(y)
                    bboxes.append(bbox)
                    bboxes_avoid.append(bbox_avoid)
                    bboxes_enforce.append(bbox_enforce)
        X = np.concatenate(data, axis=0)
        y = np.concatenate(labels, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        bboxes_avoid = np.concatenate(bboxes_avoid, axis=0)
        bboxes_enforce = np.concatenate(bboxes_enforce, axis=0)
        assert(X.shape[0] == bboxes.shape[0])
        return TextColorDataset(X, y), bboxes, bboxes_avoid, bboxes_enforce

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fname = os.path.join(save_dir, 'textcolor_corr%0.2f.npz'%exp_no)
    if os.path.exists(fname):
        print("Loading from cached data")
        tmp = np.load(open(fname, 'rb'), allow_pickle=True)
        train_data = TextColorDataset(tmp['x_train'], tmp['y_train'])
        test_data = TextColorDataset(tmp['x_test'], tmp['y_test'])
        train_coords = tmp['train_coords']
        test_coords = tmp['test_coords']
        train_avoid = tmp['train_avoid']
        test_avoid = tmp['test_avoid']
        train_enforce = tmp['train_enforce']
        test_enforce = tmp['test_enforce']
    else:
        print("Generating from scratch")
        train_data, train_coords, train_avoid, train_enforce = create_dataset(n=n)
        test_data, test_coords, test_avoid, test_enforce = create_dataset(n=400)

    if save and not os.path.exists(fname):
        np.savez(open(fname, 'wb'), 
                 x_train=train_data.X, 
                 y_train=train_data.y, 
                 x_test=test_data.X, 
                 y_test=test_data.y, 
                 train_coords=train_coords, 
                 test_coords=test_coords,
                 train_avoid=train_avoid,
                 test_avoid=test_avoid,
                 train_enforce=train_enforce,
                 test_enforce=test_enforce)

    return train_data, test_data, train_coords, test_coords, train_avoid, test_avoid, train_enforce, test_enforce
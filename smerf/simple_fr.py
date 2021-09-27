from .textbox_data import *
from .eval import *
import matplotlib.pyplot as plt

### EXP1.11 Full reliance : random white sticker location
def generate_textbox_data(n=10000, save=True, save_dir='data', exp_no=1.11, random_bg=False):
    """
    Generate textbox data
    
    :param n: number of training data points per bucket
    :param save: flag for saving the generated data points
    :param save_dir: directory to save the generated data points
    :exp_no: experiment number
    :random_bg: flag for random background
    
    :return training/test data, and primary/secondary regions in the images to focus/avoid.   
    """
    def create_data(patch, text, switch, n=n):
        """
        Given a set of features to include/exclude, generate images
        
        :param patch: feature for the larger Box in the image
        :param text: feature for the text in the image
        :param switch: feature for the smaller Box in the image
        
        :return data, label, coordinates for the features in the image 
        """
        print('Creating %d images with SWITCH=%d / PATCH=%d / TEXT=%d...'%(n, switch, patch, text))

        def create_single_img(switch, patch, text):
            """
            Create a single image file consisting of the features specified in the arguments.
    
            :return image, label, and coordinates of where each features are located in the image.
            """
            feature = np.zeros((6))
            feature[0] = text # -1 = no text, 0 = A, 1 = B
            feature[1] = np.random.uniform() #x
            feature[2] = np.random.uniform() #y
            feature[3] = 3 # character color white
            if random_bg:
                feature[4] = -3 # random gray background
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
                # convert the features into an image
                im, dim, sw = vec2im(feature, x_start=x_start, y_start=y_start, switch=switch, s_color=(255,255,255), p_color=(255,255,255))
                x_end = x_start+10
                y_end = y_start+10
            else:
                # convert the features into an image
                im, dim, sw = vec2im(feature, switch=switch, s_color=(255, 255, 255), p_color=(255,255,255))
                x_start = None
                y_start = None
                x_end = None
                y_end = None

            # Specify the label according to the model reasoning desired
            label = patch
            
            # Return the image, label, location of the object on which the model decision is based on, location(s) of other objects not relevant
            return np.float32(im / 255), label, (y_start, y_end, x_start, x_end), dim, sw

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

    def create_dataset(n=n):
        """
        Generate n images that consist of the features above.
        """
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
    if os.path.exists(fname): # if the file exists, load from the data directory
        print("Loading from cached data")
        train_data, test_data, train_primary, train_secondary, test_primary, test_secondary = \
            load_data(exp_no, save_dir)
    else: # otherwise generate a new dataset from scratch
        print('Generating data from scratch')
        train_data, train_coord, train_avoid, train_avoid2 = create_dataset(n=n)
        test_data, test_coord, test_avoid, test_avoid2 = create_dataset(n=500)
        train_data, test_data, train_primary, train_secondary, test_primary, test_secondary = \
            save_data(exp_no, save_dir, train_data, test_data, train_coord, train_avoid, \
                      train_avoid2, test_coord, test_avoid, test_avoid2, save=save) 
        
    return train_data, test_data, train_primary, test_primary, train_secondary, test_secondary
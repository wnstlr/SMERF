import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import os
from smerf.eval import setup_bboxes
import pickle

DATA_DIR = '../data/'

class TextBoxDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Interpolate between blue (v = 0) and red (v = 1)
def shade(im, v):
    if v == -1:
        im[:, :, :] = 255 #plain white background
    elif v == -2:
        im[:,:,:] = 0 # plain black background
    elif v == -3:
        im[:,:,:] = np.asarray(np.random.random((64,64,3)) * 100, dtype=int) # random gray background
    elif v == -4:
        # natural image background
        places_img_file = pickle.load(open(os.path.join(DATA_DIR, 'places_img_file.pkl'), 'rb'))
        choices = places_img_file['stadium/baseball']
        img_ids = [0, 9, 10, 12, 15, 16, 17, 19, 20, 21, 24, 25, 26, 27, 28, 33, 34, 35, 37, 39, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 55, 56, 58, 64, 65, 58, 68, 71, 74, 78, 86, 88, 90, 91, 92, 93]
        #choices = places_img_file['bamboo_forest'] 
        #img_ids = [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 25, 28, 37, 44, 47, 57, 59, 65, 68, 69, 72, 75, 77, 85, 93, 96, 98, 99]
        img_dir = os.path.join(DATA_DIR, 'val_256')
        img = Image.open(os.path.join(img_dir, choices[np.random.choice(img_ids)]))
        img = img.resize((64,64))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.7)
        img = np.array(img)
        im[:,:,:] = img
    else:
        im[:, :, 0] = 255 * v
        im[:, :, 2] = 255 * (1 - v)
    return im

# Add a square
def sticker(im, x_start = 0, y_start = 0, delta = 10, color = [0, 0, 0]):
    im[y_start:y_start + delta, x_start:x_start + delta, :] = color
    return im

# Add text
def text(im, text, x, y, color = (0,0,0), size = 20):
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("../data/arial.ttf", size)
    w, h = font.getsize(text)
    draw.text((x, y), text, color, font = font)
    im = np.array(im)
    return im, (w,h)

def vec2im(features, **kwargs):
    """
    Convert a feature into an image with certain features.
    
    :return image, text location, small box location
    """
    im = np.zeros((64, 64, 3), dtype = np.uint8)
        
    im = shade(im, features[4])
    
    if features[5] == 1: # place large box in the image
        if 'x_start' in kwargs.keys():
            patch_x = kwargs['x_start']
        else:
            patch_x = 0
        if 'y_start' in kwargs.keys():
            patch_y = kwargs['y_start']
        else:
            patch_y = 0
        if 'p_color' in kwargs.keys():
            p_color = kwargs['p_color']
        else:
            p_color = [0,0,0]

        if 'p_size' in kwargs.keys():
            p_delta = kwargs['p_size']
        else:
            p_delta = 10
        # Add a large box in the image at the location set in the argument.
        im = sticker(im, x_start=patch_x, y_start=patch_y, color=p_color, delta=p_delta)
    else: # no large box in the image so the locations are set to None
        patch_x = None
        patch_y = None
    
    # Determine the character to be included in the image
    if features[0] == 0:
        char = "A"
    elif features[0] == 1:
        char = "B"
    elif features[0] == -1:
        char = None # no character
    
    # Determine the color of the character to be included in the image
    if features[3] == 0: # set text color as  black
        color = (0, 0, 0)
    elif features[3] == 1: # set text color as green
        color = (0, 255, 0)
    elif features[3] == 3: # set text color as white
        color = (255, 255, 255)
    elif features[3] == 2: # set manual text color from (R, G, B) input
        color = kwargs['color']
    textloc = (None, None, None, None)
    
    # Add text if character is not None
    if char != None:
        xstart = int(36 * features[1] + 6)
        ystart = int(36 * features[2] + 6)
        im, dim = text(im, char, xstart, ystart, color = color)
        # keep the location of the character to return
        textloc = (ystart, ystart+dim[1], xstart, xstart+dim[0])
    
    # Add a small box if switch argument is turned on
    dist_ = 300
    if kwargs['switch'] == 1: # small box at a fixed location
        switch_x = 58
        switch_y = 58
        im = sticker(im, x_start=58, y_start=58, delta=4, color=kwargs['s_color'])
    elif kwargs['switch'] == 2: # small box at a random location
        while True:
            switch_x = np.random.random_integers(0, 53)
            switch_y = np.random.random_integers(0, 53)
            # prevent overlap of switch with text and larger box
            if patch_x is not None and features[0] != -1: # yes patch, yes character
                if (int(36*features[1]) - switch_x)**2 + (int(36*features[2]) - switch_y)**2 > dist_+100 and \
                    (patch_x - switch_x)**2 + (patch_y - switch_y)**2 > dist_:
                    break
            elif patch_x is None and features[0] != -1: # no patch, yes character
                if (int(36*features[1]) - switch_x)**2 + (int(36*features[2]) - switch_y)**2 > dist_+100:
                    break 
            elif patch_x is not None and features[0] == -1: # yes patch, no character
                if (patch_x - switch_x)**2 + (patch_y - switch_y)**2 > dist_:
                    break
            elif patch_x is None and features[0] == -1: # neither
                break
        im = sticker(im, x_start=switch_x, y_start=switch_y, delta=4, color=kwargs['s_color'])
    else:
        switch_x = None
        switch_y = None
    
    # keep the small box location to return
    if switch_x is not None:
        switch_loc = (switch_y, switch_y+4, switch_x, switch_x+4)
    else:
        switch_loc = (None, None, None, None)
        
    # return the image generated, character location, and small box location
    return im, textloc, switch_loc

def save_data(exp_no, save_dir, train_data, test_data, train_coord, train_avoid, train_avoid2, test_coord, test_avoid, test_avoid2, save=True):
    # setup bbox info to save to the file
    fname = os.path.join(save_dir, 'textbox_%0.2f.npz'%exp_no)
    # NOTE need to specify below based on different type of experiments
    if exp_no in [1.11, 2.11]: # for simple FR and NR, only one object to include
        gt_flag = [1,0,0]
    elif exp_no == 1.2: # for complex-FR, there are two ground-truth objects to include
        gt_flag = [1,0,1]
    elif exp_no >= 3.7: # for complex-CR, there are two ground-truth objects to inlcude
        gt_flag = [1,0,1]
    train_primary, train_secondary = setup_bboxes(train_coord, train_avoid, train_avoid2, np.array(range(train_data.X.shape[0])), gt_flag=gt_flag)
    test_primary, test_secondary = setup_bboxes(test_coord, test_avoid, test_avoid2, np.array(range(test_data.X.shape[0])), gt_flag=gt_flag)
    if save:
        np.savez(open(fname, 'wb'),
                    x_train=train_data.X, 
                    y_train=train_data.y, 
                    x_test=test_data.X, 
                    y_test=test_data.y,  
                    train_primary=train_primary,
                    test_primary=test_primary,
                    train_secondary=train_secondary,
                    test_secondary=test_secondary)
    return train_data, test_data, train_primary, train_secondary, test_primary, test_secondary
    
def load_data(exp_no, load_dir): 
    fname = os.path.join(load_dir, 'textbox_%0.2f.npz'%exp_no)
    tmp = np.load(open(fname, 'rb'), allow_pickle=True)
    train_data = TextBoxDataset(tmp['x_train'], tmp['y_train'])
    test_data = TextBoxDataset(tmp['x_test'], tmp['y_test'])
    train_primary = tmp['train_primary']
    test_primary = tmp['test_primary']
    train_secondary = tmp['train_secondary']
    test_secondary = tmp['test_secondary']
    return train_data, test_data, train_primary, train_secondary, test_primary, test_secondary
     
# Generate text data with spurious features
# make the labels to be correlated with color, not the digit itself
# or the other way
def sample_uniform():
    feature = np.zeros((6))
    feature[0] = np.random.randint(2) #character
    feature[1] = np.random.uniform() #x
    feature[2] = np.random.uniform() #y
    feature[3] = 0
    feature[4] = np.random.uniform() # shade
    feature[5] = 0
    return feature

def generate_data(n=10000):
    #plain data
    rep = np.zeros((n, 6))
    labels = np.zeros(n)
    im = np.zeros((n, 64, 64, 3))
    for i in range(n):
        rep[i] = sample_uniform()
        im[i] = vec2im(rep[i])
        labels[i] = int(rep[i][0])
    im = np.float32(im / 255)
    return im, labels, rep

def original_textbox_data(n=10000, save=True, save_dir='data'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    fname = os.path.join(save_dir, 'textbox_original.npz')
    if os.path.exists(fname):
        tmp = np.load(open(fname, 'rb'))
        im = tmp['x_train']
        labels = tmp['y_train']
        im_test = tmp['x_test']
        labels_test = tmp['y_test']
    else:
        # train data
        im, labels, rep = generate_data(n=n)
        # val data
        test_n = int(n * 0.3)
        im_test, labels_test, rep_test = generate_data(n=test_n)

    train_data = TextBoxDataset(im, labels)
    test_data = TextBoxDataset(im_test, labels_test)

    if save:
        np.savez(open(fname, 'wb'), x_train=train_data.X, y_train=train_data.y, x_test=test_data.X, y_test=test_data.y)
        np.savez(open(os.path.join(save_dir, 'textbox_original_meta.npz'), 'wb'), rep=rep, rep_test=rep_test)

    return train_data, test_data

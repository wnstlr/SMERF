import os
import pickle

DATA_DIR = '../data'
places_id_file = os.path.join(DATA_DIR, 'filelist_places365-standard/categories_places365.txt')
places_id = dict()
with open(places_id_file, 'r') as f:
    for l in f.readlines():
        s = l.split(' ')
        places_id[int(s[1])] = s[0][3:]

places_val_file = os.path.join(DATA_DIR, 'filelist_places365-standard/places365_val.txt')
places_img_file = dict()
for k in places_id.keys():
    places_img_file[places_id[k]] = []
with open(places_val_file, 'r') as f:
    for l in f.readlines():
        s = l.split(' ')
        places_img_file[places_id[int(s[1])]].append(s[0])
        
pickle.dump(places_img_file, open(os.path.join(DATA_DIR, 'places_img_file.pkl', 'wb')))
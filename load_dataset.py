import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing import image

def load_dataset(num_samples=20000):
    # get list
    dataset_dir = "Data/png/"
    filelist_path = dataset_dir + "filelist.txt"
    filelist = []
    with open(filelist_path) as infile:
        for line in infile.readlines():
            filelist.append(line.strip())

    # get images and labels
    images =  []
    labels = []
    count = 0
    for filepath in filelist[:num_samples]:
        # parse label until slash and add to list
        labels.append(filepath.partition("/")[0])

        # load image
        image_path = dataset_dir + filepath
        img = image.load_img(image_path,target_size=(224, 224,3))
        img = image.img_to_array(img)

        # display progress
        if count %100 == 0:
            print(count, "/", 20000)
        images.append(img)
        count += 1

    return np.asarray(images), pd.DataFrame(labels)

def train_test_val_1hot(images,labels):

    # train test split
    xtrain, xtest,ytrain, ytest = train_test_split(images,labels,test_size=.2)
    xtrain, xval,ytrain, yval = train_test_split(xtrain,ytrain,test_size=.2)

    # one hot encode with sklearn
    enc = OneHotEncoder(handle_unknown = 'ignore')

    # ONLY FIT TO TRAINING DATA.
    # If unknown category shows up in validation, shouldn't light any up.
    enc.fit(ytrain)

    ytrain_1hot = enc.transform(ytrain).toarray()
    yval_1hot = enc.transform(yval).toarray()
    ytest_1hot = enc.transform(ytest).toarray()

    return xtrain,xval, xtest,ytrain_1hot,yval_1hot, ytest_1hot

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    images_train =  []
    images_test = []
    labels_train = []
    labels_test = []
    count = 0
    for filepath in filelist[:num_samples]:
        # load image
        image_path = dataset_dir + filepath
        img = image.load_img(image_path,target_size=(224, 224,3))
        img = image.img_to_array(img)

        # add to list
        if count %80 < 60:
            labels_train.append(filepath.partition("/")[0])
            images_train.append(img)
        else:
            labels_test.append(filepath.partition("/")[0])
            images_test.append(img)

        # display progress
        if count %100 == 0:
            print(count, "/", 20000)
        count += 1
    return np.asarray(images_train), pd.DataFrame(labels_train), \
            np.asarray(images_test), pd.DataFrame(labels_test)

def onehot(ytrain, ytest):
    # one hot encode with sklearn
    enc = OneHotEncoder(handle_unknown = 'ignore')

    # ONLY FIT TO TRAINING DATA.
    # If unknown category shows up in validation, shouldn't light any up.
    enc.fit(ytrain)

    ytrain_1hot = enc.transform(ytrain).toarray()
    ytest_1hot = enc.transform(ytest).toarray()

    return ytrain_1hot, ytest_1hot, enc.categories_[0]

def stdScale(xtrain, xval, xtest):
    # scaler = StandardScaler()
    # xtrain = scaler.fit_transform( xtrain )
    # xtest = scaler.transform( xtest )

    return xtrain/255, xval/255, xtest/255

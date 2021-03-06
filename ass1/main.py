import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.python.client import device_lib

from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Activation, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD

# silence warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# local imports
from load_dataset import *

# get args from command line
flags = tf.compat.v1.flags

# dataset options
flags.DEFINE_boolean("load_dataset",False,"Whether to save preprocessed dataset")
flags.DEFINE_boolean("save_dataset",False,"Whether to save preprocessed dataset")
flags.DEFINE_string("path_prep","Data/prep/","Number of samples")

#save/load model options
flags.DEFINE_string("output_dir","output/","output directory")
flags.DEFINE_string("model","model_v3","Which model?")

# training options
flags.DEFINE_integer("num_samples",20000,"Number of samples")
flags.DEFINE_integer("epoch1",1,"Epochs for first pass")
flags.DEFINE_integer("epoch2",1,"Epochs for second pass")

#gpu options
flags.DEFINE_boolean("ubu",False,"Whether on ubuntu linux box")


opt = flags.FLAGS

def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap='Blues',output_path='.',):

    """
    Logs and plots a confusion matrix, e.g. text and image output.

    Adapted from:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        tag = '_norm'
        print("Normalized confusion matrix:")
    else:
        tag = ''
        print('Confusion matrix:')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'{output_path}/confusion{tag}.png')
    plt.close()

def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def train_v1(xtrain,ytrain,xval, yval,xtest,ytest):
    # preprocess
    xtrain, xval, xtest = stdScale(xtrain, xval, xtest)

    # define model
    input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
    num_classes = ytrain.shape[1]
    print("num_classes", num_classes)
    model = InceptionV3(input_tensor=input_tensor, weights=None, include_top=True, classes=num_classes)

    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
    )

    # Distribute the neural network over multiple GPUs if available.
    gpu_count = len(available_gpus())
    if gpu_count > 1:
        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
    else:
        print("\n\nModel not parallelized over GPUs.\n\n")
        parallel_model = model

    parallel_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # create a checkpoint to save the model
    csv_logger = keras.callbacks.CSVLogger(
        opt.output_dir + "history_" + opt.model + ".csv",
    )

    # train
    print("ephochs2:",opt.epoch2)
    parallel_model.fit(xtrain,
                       ytrain,
                       validation_data=(xval,yval),
                       epochs=opt.epoch2,
                       callbacks=[csv_logger])

    # parallel_model.save(opt.output_dir + opt.model + ".h5")

    results2(xtrain,ytrain,xval, yval,xtest,ytest, parallel_model)

def train_v2(xtrain,ytrain,xval, yval,xtest,ytest):
    # preprocess
    xtrain = preprocess_input(xtrain)
    xval = preprocess_input(xval)
    xtest = preprocess_input(xtest)
    extractor_model = InceptionV3(weights='imagenet', include_top=False)

    xtrain = extractor_model.predict(xtrain)
    xval = extractor_model.predict(xval)
    xtest = extractor_model.predict(xtest)

    # linear softmax model
    num_categories = ytrain.shape[1]
    input_shape = xtrain.shape[1:]

    inputs = Input(shape=input_shape)                 # input layer
    flat = Flatten()(inputs)
    outputs = Dense(num_categories, activation='softmax')(flat) # output layer
    model = Model(inputs, outputs)

    # For a multi-class classification problem
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # create a checkpoint to save the model history
    csv_logger = keras.callbacks.CSVLogger(
        opt.output_dir + "history_" + opt.model + ".csv",
    )

    # train
    print("ephochs2:",opt.epoch2)
    model.fit(xtrain,
                       ytrain,
                       validation_data=(xval,yval),
                       epochs=opt.epoch2,
                       callbacks=[csv_logger])

    results2(xtrain,ytrain,xval, yval,xtest,ytest, model)


def train_v3(xtrain,ytrain,xval, yval,xtest,ytest):
    # preprocess
    xtrain = preprocess_input(xtrain)
    xval = preprocess_input(xval)
    xtest = preprocess_input(xtest)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    num_categories = ytrain.shape[1]
    predictions = Dense(num_categories, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
    )

    # train the model on the new data for a few epochs
    print("ephochs1:",opt.epoch1)
    history = model.fit(xtrain,
                        ytrain,
                        validation_data=(xval,yval),
                        epochs=opt.epoch1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # we chose to train the top 2 inception blocks. we will freeze the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Distribute the neural network over multiple GPUs if available.
    gpu_count = len(available_gpus())
    if gpu_count > 1:
        print(f"\n\nModel parallelized over {gpu_count} GPUs.\n\n")
        parallel_model = keras.utils.multi_gpu_model(model, gpus=gpu_count)
    else:
        print("\n\nModel not parallelized over GPUs.\n\n")
        parallel_model = model

    parallel_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    # create a checkpoint to save the history
    csv_logger = keras.callbacks.CSVLogger(
        opt.output_dir + "history_" + opt.model + ".csv",
    )


    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print("ephochs2:",opt.epoch2)
    parallel_model.fit(xtrain,
              ytrain,
              validation_data=(xval,yval),
              epochs=opt.epoch2,
              callbacks=[csv_logger])

    results2(xtrain,ytrain,xval, yval,xtest,ytest, parallel_model)

def results2(xtrain,ytrain,xval, yval,xtest,ytest,model):

    print("Val:")
    preds = model.predict(xval)
    print(
            classification_report(
                np.argmax(yval, axis=-1),
                np.argmax(preds, axis=-1),
            )
    )

    print("Test:")
    preds = model.predict(xtest)
    print(
            classification_report(
                np.argmax(ytest, axis=-1),
                np.argmax(preds, axis=-1),
            )
    )

if __name__ == '__main__':
    # configure gpus if on ubuntu
    if opt.ubu == True:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4624)])
        pass

    # load data
    if opt.load_dataset == False:
        xtrain,ytrain,xtest,ytest = load_dataset(num_samples=opt.num_samples)

        # one hot
        ytrain,ytest,target_names = onehot(ytrain,ytest)
        # split off validation
        xtrain, xval,ytrain, yval = train_test_split(xtrain,ytrain,test_size=.2)

        if opt.save_dataset == True:
            pickle.dump((xtrain,ytrain), open(opt.path_prep+"train.p", "wb"), protocol=4)
            pickle.dump((xval, yval), open(opt.path_prep+"val.p", "wb"), protocol=4)
            pickle.dump((xtest, ytest), open(opt.path_prep+"test.p", "wb"), protocol=4)
            pickle.dump(target_names, open(opt.path_prep+"target_names.p", "wb"), protocol=4)
    else:
        xtrain,ytrain = pickle.load(open(opt.path_prep + "train.p","rb"))
        xval,yval = pickle.load(open(opt.path_prep + "val.p","rb"))
        xtest,ytest = pickle.load(open(opt.path_prep + "test.p","rb"))
        target_names = pickle.load(open(opt.path_prep + "target_names.p","rb"))

    #train model
    if opt.model == "model_v1":
        train_v1(xtrain,ytrain,xval, yval,xtest,ytest)
    if opt.model == "model_v2":
        train_v2(xtrain,ytrain,xval, yval,xtest,ytest)
    if opt.model == "model_v3":
        train_v3(xtrain,ytrain,xval, yval,xtest,ytest)

import pickle

import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# local imports
from load_dataset import *

# get args from command line
flags = tf.app.flags
flags.DEFINE_boolean("load_dataset",False,"Whether to save preprocessed dataset")
flags.DEFINE_boolean("save_dataset",False,"Whether to save preprocessed dataset")
flags.DEFINE_string("path_prep","Data/prep/prep.p","Number of samples")

flags.DEFINE_integer("num_samples",20000,"Number of samples")
flags.DEFINE_integer("epoch1",1,"Epochs for first pass")
flags.DEFINE_integer("epoch2",1,"Epochs for second pass")
opt = flags.FLAGS
print("num_samples:",opt.num_samples)


def train(xval,yval):
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
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model on the new data for a few epochs
    history = model.fit(xtrain,
                          ytrain,
                          validation_data=(xval,yval),
    #                       batch_size=16,
                          epochs=opt.epoch1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history = model.fit(xtrain,
                          ytrain,
                          validation_data=(xval,yval),
    #                       batch_size=16,
                          epochs=opt.epoch2)

if __name__ == '__main__':
    # load data
    if opt.load_dataset == False:
        images,labels = load_dataset(num_samples=opt.num_samples)
        # preprocess
        images_prep = preprocess_input(images)

        # split
        xtrain,xval,xtest, ytrain,yval,ytest = train_test_val_1hot(images_prep,labels)

        if opt.save_dataset == True:
            pickle.dump((xtrain,xval,xtest, ytrain,yval,ytest), open(opt.path_prep, "wb"))
    else:
        xtrain,xval,xtest, ytrain,yval,ytest = pickle.load(open(opt.path_prep,"rb"))

    #train model
    train(xval,yval)

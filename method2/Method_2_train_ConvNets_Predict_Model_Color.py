import os
from math import ceil

GPUS = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS


import numpy as np
import tensorflow as tf

from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD



#from utils import generator_batch_multitask
from generator import ModelColorGererator




np.random.seed(1024)

FINE_TUNE = False
LEARNING_RATE = 0.001
NBR_EPOCHS = 3
BATCH_SIZE = 64
IMG_WIDTH = 299
IMG_HEIGHT = 299
monitor_index = 'loss'
NBR_MODELS = 250
NBR_COLORS = 7
RANDOM_SCALE = True
nbr_gpus = len(GPUS.split(','))
INITIAL_EPOCH = 0



train_path = 'train_valid/train_image_model_color.txt'
val_path = 'train_valid/valid_image_model_color.txt'

save_model_path = "./models/InceptionV3_vehicleModelColor.h5"

if __name__ == "__main__":

    if FINE_TUNE:
        print('Finetune and Loading InceptionV3 Weights ...')
        model = load_model(save_model_path)
        INITIAL_EPOCH = 0

    else:
        print('Loading InceptionV3 Weights from ImageNet Pretrained ...')

        inception = InceptionV3(include_top=False, weights= None,
               input_tensor=None, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling = 'avg')


        f_base = inception.get_layer(index = -1).output     # shape=(None, 1, 1, 2048)

        f_acs = Dense(1024, name='f_acs')(f_base)

        f_model = Dense(NBR_MODELS, activation='softmax', name='predictions_model')(f_acs)

        f_color = Dense(NBR_COLORS, activation='softmax', name='predictions_color')(f_acs)

        model = Model(outputs = [f_model, f_color], inputs = inception.input)



    print('Training model begins...')

    optimizer = SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 0.0, nesterov = True)

    
    # 未实现tensorflow的多gpu    
    print('Using a single GPU.\n')
        
    model.compile(loss=["categorical_crossentropy", "categorical_crossentropy"],
                  #loss_weights = [0.6, 0.4],
                  optimizer=optimizer, metrics=["accuracy"])

    #model.summary()

    best_model_saved = "./models/InceptionV3_vehicleModelColor.h5"
    # Define several callbacks

    checkpoint = ModelCheckpoint(best_model_saved, verbose = 1)

    reduce_lr = ReduceLROnPlateau(monitor='val_'+monitor_index, factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)

    early_stop = EarlyStopping(monitor='val_'+monitor_index, patience=15, verbose=1)

    train_data_lines = open(train_path).readlines()
    # Check if image path exists.
    train_data_lines = [w for w in train_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_train = len(train_data_lines)
    print('# Train Images: {}.'.format(nbr_train))
    steps_per_epoch = int(ceil(nbr_train * 1. / BATCH_SIZE))

    val_data_lines = open(val_path).readlines()
    val_data_lines = [w for w in val_data_lines if os.path.exists(w.strip().split(' ')[0])]
    nbr_val = len(val_data_lines)
    print('# Val Images: {}.'.format(nbr_val))
    validation_steps = int(ceil(nbr_val * 1. / BATCH_SIZE))

    model.fit_generator(ModelColorGererator(train_data_lines,
                        nbr_class_model = NBR_MODELS, nbr_class_color = NBR_COLORS,
                        batch_size = BATCH_SIZE, img_width = IMG_WIDTH,
                        img_height = IMG_HEIGHT, random_scale = RANDOM_SCALE,
                        shuffle = True, augment = True),
                        steps_per_epoch = steps_per_epoch, epochs = NBR_EPOCHS, verbose = 1,
                        validation_data = ModelColorGererator(val_data_lines,
                        nbr_class_model = NBR_MODELS, nbr_class_color = NBR_COLORS, batch_size = BATCH_SIZE,
                        img_width = IMG_WIDTH, img_height = IMG_HEIGHT,
                        shuffle = False, augment = False),
                        validation_steps = validation_steps,
                        callbacks = [checkpoint, reduce_lr], initial_epoch = INITIAL_EPOCH,
                        max_queue_size = 80, workers = 8, use_multiprocessing=True)

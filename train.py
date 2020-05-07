# -*- coding: utf-8 -*-

import os
from tensorflow.compat.v1.keras.models import Model
from tensorflow.compat.v1.keras.layers import Dense, Dropout
from tensorflow.compat.v1.keras.applications.mobilenet import MobileNet
from tensorflow.compat.v1.keras.optimizers import SGD
from tensorflow.compat.v1.keras.losses import categorical_crossentropy
from data_loader import generator, dataset_size
import logging
import math

def train_new_model():

    print('Start creation of new model')
    base_model_class_name = MobileNet
    base_model = base_model_class_name(input_shape=(224, 224, 3),
                                       include_top=False,
                                       weights='imagenet', 
                                       pooling='avg')
    
    for layer in base_model.layers:
        layer.trainable = True

    print('The new final layer is added')    
    d_layer_name = "new_dense_layer"   
    x = Dropout(0.75)(base_model.output)
    x = Dense(10,
              activation='softmax',
              name=d_layer_name)(x)
    model = Model(base_model.input,
                  x)
    
    optimizer = SGD(lr=2e-3,
                    momentum=0.9,
                    nesterov=True)
    
    loss_function = categorical_crossentropy
    
    model.compile(optimizer,
                  loss=loss_function,
                  metrics=['accuracy'])

    print('Model is ready\n')
    
    return model
   
def main():
    dataset_training_size = dataset_size(train = True)
    dataset_val_size = dataset_size(val = True)
    
    loaded_epoch = 0
    epochs_to_train = 5
    
    print('\nStarting epoch: {}\nEpochs to train: {}'.format(loaded_epoch,
                                                             epochs_to_train))
    
    
    model = train_new_model()
    
    batchsize = 16
    print('Batchsize: {}'.format(batchsize))    
    
    training_steps = math.ceil(dataset_training_size / batchsize)
    val_steps = math.ceil(dataset_val_size / batchsize)    
    
    train_generator = generator(batchsize=batchsize,
                                epochs = epochs_to_train,
                                epoch_steps = training_steps, 
                                train=True, 
                                shuffle=True)
    
    val_generator = generator(batchsize=batchsize,
                              epochs = epochs_to_train,
                              epoch_steps = val_steps,
                              val=True)
    
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=training_steps,
                        epochs=epochs_to_train+loaded_epoch,
                        verbose=1,
                        initial_epoch=loaded_epoch,
                        validation_data=val_generator,
                        validation_steps=val_steps)
    
if __name__ == "__main__":
    logging.disable(logging.WARNING)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    print('\nStart train_model')
    
    main()
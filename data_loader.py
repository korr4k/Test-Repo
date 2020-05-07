# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.applications.mobilenet as mobilenet


train_scores = np.load('train_scores.npy')
val_scores = np.load('val_scores.npy')
test_scores = np.load('test_scores.npy')
train_paths = np.load('train_paths.npy')
val_paths = np.load('val_paths.npy')
test_paths = np.load('test_paths.npy')

def dataset_size(train = False, val = False, test = False):
    if (train):
        return len(train_paths)
    elif(val):
        return len(val_paths)
    elif(test):
        return len(test_paths)

def dataset_scores(train = False, val = False, test = False):
    if (train):
        return train_scores
    elif(val):
        return val_scores
    elif(test):
        return test_scores

def dataset_paths(train = False, val = False, test = False):
    if (train):
        return train_paths
    elif(val):
        return val_paths
    elif(test):
        return test_paths
    
def image_gen(ds_path, ds_scores=None):
    for i, path in enumerate(ds_path):        
        img_to_numpy = np.load(path + '.npy')
        
        if (ds_scores is not None):
            yield img_to_numpy, ds_scores[i]
        else:
            yield img_to_numpy

def pre_processing_image(image, score=None):
    image = mobilenet.preprocess_input(image)
        
    if score is None:
        return image
    else:
        return image, score
        
def generator(batchsize, epochs, epoch_steps, train=False, val=False, test=False, shuffle=False):
    with tf.Session() as sess:      
        if (train):
            dataset = tf.data.Dataset.from_generator(lambda: image_gen(train_paths, train_scores),
                                                      output_types=(tf.float32, tf.float32))
        elif(val):
            dataset = tf.data.Dataset.from_generator(lambda: image_gen(val_paths, val_scores),
                                                      output_types=(tf.float32, tf.float32))
        else:
            dataset = tf.data.Dataset.from_generator(lambda: image_gen(test_paths),
                                                      output_types=(tf.float32))       

        if (shuffle and train):
            dataset = dataset.shuffle(buffer_size=dataset_size(train=True)) 
        elif (shuffle and val):
                dataset = dataset.shuffle(buffer_size=dataset_size(val=True)) 
        elif (shuffle and test):
                dataset = dataset.shuffle(buffer_size=dataset_size(test=True)) 
            
        dataset = dataset.batch(batchsize)
        dataset = dataset.map(pre_processing_image,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(1)
        dataset = dataset.repeat(count = -1)
        
        iterable = tf.data.make_initializable_iterator(dataset)
        batch = iterable.get_next()
        sess.run(iterable.initializer)
        
        while True:
            try:
                data = sess.run(batch)
                yield data
            except:
                pass

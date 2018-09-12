# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:41:45 2018

@author: Miguel
"""
import cv2
import PIL
from PIL import Image
import numpy as np
from six.moves import cPickle as pickle
from google_images_download import google_images_download
import os
import shutil
import tensorflow as tf
import imageio


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout

class ImageCollector(): 
    def __init__(self, dest,validation = 0, Query = None,  split = 0.8):
        self.images = []
        self.formated_images = []
        self.Query = Query
        self.total_removed = 0
        self.validation = validation
        
    def multiply_imgs(self, num = 4):
        Query = self.Query
        pts1 = np.float32([[30,30],[50,30],[30,50]])
        pts_array = []
        pts_array.append(np.float32([[25,30],[50,35],[30,60]]))
        pts_array.append(np.float32([[30,30],[50,30],[20,50]]))
        #pts_array.append(np.float32([[30,30],[50,30],[40,50]]))
        #pts_array.append(np.float32([[35,30],[50,32],[35,45]]))
        count = 0
        for q in Query:
            img_names = os.listdir('./downloads/{}'.format(q))
            for pic in img_names:
                try:
                    img = imageio.imread('./downloads/{}/{}'.format(q, pic))
                except:
                    pass
                for i,p in enumerate(pts_array):  
                    try:
                        M = cv2.getAffineTransform(pts1,p)                       
                        dst = cv2.warpAffine(img, M, (int(img.shape[1]*1.5), int(img.shape[0]*1.5)))                       
                        imageio.imwrite('./downloads/{}/{}'.format(q, '{}'.format(i)+pic), dst)
                        count = count+1
                    except:
                        pass

        print('Images added: {}'.format(count))
        
    def download_images(self, Query, maxN = None, MaxRes = None):
        for q in Query:
            arguments = {"keywords":"{}".format(q),"limit":maxN,"print_urls":False, 'image_directory':'./{}'.format(q), 'chromedriver':'C:\\Users\\Miguel\\Documents\\Machine learning\\Model creator\\chromedriver.exe'}
            response = google_images_download.googleimagesdownload()
            path = response.download(arguments)
    
    def filter_imgs(self):
        Query = self.Query
        c = 0
        for q in Query:
            img_names = os.listdir('./downloads/{}'.format(q))
            for img in img_names:
                if not img.lower().endswith(('.png', '.jpg')):
                    os.remove('./downloads/{}/{}'.format(q, img))
                    c = c+1
        print('Removed {} images'.format(c))
        self.total_removed = self.total_removed + c
    
    def normalize(self, img_array):
        pixel_depth = 255.0
        #return cv2.normalize(img_array, img_array, -1, 1, cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        return (img_array.astype(float)-img_array.min())/(img_array.max()-img_array.min())
    
    def pickle_images(self, split, dest): #Filling up test/validation sets firt Increased accuracy. Best images are used for testing.
        #get list of images
        Query = self.Query
        validation = self.validation
        train_fails = []
        test_fails = []
        image_paths = dict()
        total_n_images = 0
        image_counters = []
        for q in Query:
            image_counters.append(0)
            image_paths[q] = os.listdir('./downloads/{}'.format(q))
            total_n_images = total_n_images + len(image_paths[q])
        total_train = round((split-validation)*total_n_images)
        total_test = total_n_images - total_train - round(validation * total_n_images)
        total_val = total_n_images-total_train-total_test
    
        images = dict()
        if validation != 0:
            images['val_data'] =  np.ndarray(shape = (total_val, 32,32,3))
            images['val_labels'] = np.ndarray(shape = (total_val, len(Query)))
            val_fails = []
        images['train_data'] = np.ndarray(shape = (total_train, 32,32,3))
        images['test_data'] =  np.ndarray(shape = (total_test, 32,32,3))
        images['test_labels'] = np.ndarray(shape = (total_test, len(Query)))
        images['train_labels'] = np.ndarray(shape = (total_train, len(Query)))
        percent = int(total_n_images/100)

        for i in range(total_n_images): #Randomize when each set is filled. This should maximize performance
            if i%(percent*10) == 0:
                print("{}% done".format(round(i*100/total_n_images, 2)))
            fail_flag = False
            if len(Query) > 1:
                pick_query = np.random.randint(low = 0, high = len(Query))
                while(image_counters[pick_query] == len(image_paths[Query[pick_query]])):#if current list is empty, pick another
                    pick_query = np.random.randint(low = 0, high = len(Query))
            else:
                pick_query = 0
            #picking one image from the randomly selected query
            #opening the next image from the randomly selected Query and lowering it's size
            try:
                ph_image = Image.open('./downloads/{}/{}'.format(Query[pick_query], image_paths[Query[pick_query]][image_counters[pick_query]]))
            except:
                fail_flag= True
             #increasing the counter for that query
             #ph for one hot encoded data
            ph = np.zeros(shape = (len(Query)))
            ph[pick_query] = 1
            image_counters[pick_query] = image_counters[pick_query]+1
            
            if i < total_train+total_test and i >= total_test:
                if fail_flag:
                    train_fails.append(i-total_test)
                else:
                    try:
                        images['train_data'][i- total_test, :,:,:] = self.normalize(np.array(ph_image.resize((32,32)))[:,:,0:3])
                        images['train_labels'][i-total_test,:] = ph
                    except:
                        train_fails.append(i-total_test)
            elif i < total_test:
                if fail_flag:
                    test_fails.append(i)
                else:
                    try:
                        images['test_data'][i, :,:,:] = self.normalize(np.array(ph_image.resize((32,32)))[:,:,0:3])
                        images['test_labels'][i ,:] = ph
                    except:
                        test_fails.append(i )
            else:
                if fail_flag:
                    val_fails.append(i-total_train-total_test)
                else:
                    try:
                        images['val_data'][i-total_train-total_test, :,:,:] = self.normalize(np.array(ph_image.resize((32,32)))[:,:,0:3])
                        images['val_labels'][i - total_train- total_test,:] = ph
                    except:
                        val_fails.append(i-total_train-total_test)
        
        if len(train_fails) > 0:
            images['train_data'] = np.delete(images['train_data'], train_fails, 0)
            images['train_labels'] = np.delete(images['train_labels'], train_fails, 0)
            
        if len(test_fails) > 0:
            images['test_data'] = np.delete(images['test_data'], test_fails, 0)
            images['test_labels'] = np.delete(images['test_labels'], test_fails, 0)
        
        if validation != 0:
            if len(val_fails) > 0:
                images['val_data'] = np.delete(images['val_data'], val_fails, 0)
                images['val_labels'] = np.delete(images['val_labels'], val_fails, 0)
        print('100%')
        
        print('{} Images were removed'.format(len(train_fails)+len(test_fails)))
        pickle.dump(images, open('./pickled_data/{}.pickle'.format(dest), 'wb'), protocol = 4)
        self.total_removed = self.total_removed + len(train_fails)+ len(test_fails)
        self.images = images
        print('Total Images = {}'.format(len(images['train_data'])+ len(images['test_data'])))
        
    def remove_unpickled_data(self, Query):
        for q in Query:
            shutil.rmtree('./downloads/{}'.format(q))
    

      
test = ImageCollector(Query = ['airplane.jpg', 'dog.jpg', 'trees.jpg', 'frog.jpg'], split = 0.8, dest = '')
#test.download_images(maxN = 500)
test.filter_imgs()
#test.multiply_imgs(test.Query)
test.pickle_images(split = 0.9, dest = 'test1')

class Model():
    def __init__(self,epoch, num_out, hold_prob = 0.7, learning_rate = 0.0001, layers = 2, filters = [32,64, 128], name = 'test_model'):
        self.num_out = num_out
        self.hold_prob_val = hold_prob
        self.learning_rate = learning_rate
        self.layers = layers
        self.filters = filters
        self.name = name
        self.epoch = epoch
    
    def get_images(self, dest):
        f = open('./pickled_data/{}'.format(dest), 'rb')
        self.images = pickle.load(f)
    
    def next_batch(self, data, data_labels,  steps):
        if steps != len(data):
            rand_start = np.random.randint(0,len(data)-steps) 
        else:
            rand_start = 0
        image_batch = data[rand_start:(rand_start+steps)]
        image_labels = data_labels[rand_start:(rand_start+steps)]
        return image_batch, image_labels
    
    def train_layers(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.images['train_data'].shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(512, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_out))
        model.add(Activation('softmax'))

# initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        
        model.fit(self.images['train_data'], self.images['train_labels'],
              batch_size=130,
              epochs=self.epoch,
              validation_data=(self.images['test_data'], self.images['test_labels']),
              shuffle=True)
        start = time.time()
        model.predict(self.images['test_data'])
        end = time.time()
        print('{} sec to predict {} results'.format(round(end-start,3), len(self.images['test_data'])))
        #model.save('model.h5')
        self.mod = model
        return model
                    
    model = Model(80,4, layers = 2)
    model.get_images('test1.pickle')
    model.train_layers()

                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
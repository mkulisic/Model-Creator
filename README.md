# Model-Creator
Downloads and prepossesses Images from google images and then trains a model to predict on those images

Disclaimer:
This is a work in progress and results may vary. 

Usage:
The file contains two objects that contain the functionally. In the future I am to make this executable through the command line. As of now it can only be called from another script. 

“ImageCollector” Class: 
This class downloads images based on Queries given to it. The amount of images to be downloaded is specified in the in the initialization. The images will then be reduced to a 32x32 format, normalized and stored in a pickle. The order of the images is random so that it doesn’t affect training. The user may choose split and validation values to split the data into train, test and validation sets. Final pickle consists of an object with the classes: 
‘train_data’ : Contains image data for the train set with the format [number of train images, 32, 32, 3]
‘test_data’ : Contains image data for the test set with the format [number of test images, 32, 32, 3]
‘val_data’ : Contains image data for the train set with the format [number of validation images, 32, 32, 3]
‘train_labels’: Contains sparse matrices based on the queries given with the format [number of train images, number of queries]
‘test_labels’: Contains sparse matrices based on the queries given with the format [number of test images, number of queries]
‘val_labels’: Contains sparse matrices based on the queries given with the format [number of validation images, number of queries]

Methods: 
download_images(self, Query, maxN = None, MaxRes = None):: Downloads the amount of images specified into the downloads folder. 
filter_imgs(): Removes all unusable images from the folder.
multiply_imgs(self, num = 4): Uses affine transformations to increase the number of images in the downloads folder. Still not fully working so it might be best not to use it.
pickle_images(self, split, dest): Resizes, normalizes, randomizes and separates data into train, test and validation sets. 

Ex: 
test = ImageCollector(Query = ['airplane.jpg', 'dog.jpg', 'trees.jpg', 'frog.jpg'], split = 0.8, dest = ‘’)
test.download_images(maxN = 500)
test.filter_imgs() 
#test.multiply_imgs(test.Query)
test.pickle_images(split = 0.9, dest = 'test1')

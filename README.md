**Disclaimer:**

This is a work in progress and results may vary.

**Usage:**

The file contains two objects that contain the functionally. In the future I am to make this executable through the command line. As of now it can only be called from another script.

**&quot;ImageCollector&quot; Class:**

This class downloads images based on Queries given to it. The amount of images to be downloaded is specified in the in the initialization. The images will then be reduced to a 32x32 format, normalized and stored in a pickle. The order of the images is random so that it doesn&#39;t affect training. The user may choose split and validation values to split the data into train, test and validation sets. Final pickle consists of an object with the classes:

- -- **&#39;train\_data&#39;** : Contains image data for the train set with the format [number of train images, 32, 32, 3]
- -- **&#39;test\_data&#39;** : Contains image data for the test set with the format [number of test images, 32, 32, 3]
- -- **&#39;val\_data&#39;** : Contains image data for the train set with the format [number of validation images, 32, 32, 3]
- -- **&#39;train\_labels**&#39;: Contains sparse matrices based on the queries given with the format [number of train images, number of queries]
- -- **&#39;test\_labels&#39;** : Contains sparse matrices based on the queries given with the format [number of test images, number of queries]
- -- **&#39;val\_labels&#39;** : Contains sparse matrices based on the queries given with the format [number of validation images, number of queries]

**Methods:**

- **--**** download\_images(self, Query, maxN = None, MaxRes = None)::** Downloads the amount of images specified into the downloads folder.
- **--**** filter\_imgs():** Removes all unusable images from the folder.
- **--**** multiply\_imgs(self, num = 4):** Uses affine transformations to increase the number of images in the downloads folder. Still not fully working so it might be best not to use it.
- **--**** pickle\_images(self, split, dest):** Resizes, normalizes, randomizes and separates data into train, test and validation sets.
- **--**** \_\_init\_\_(self, dest,validation = 0, Query = None,  split = 0.8)**

**Ex:**

    test = ImageCollector(Query = [&#39;airplane.jpg&#39;, &#39;dog.jpg&#39;, &#39;trees.jpg&#39;, &#39;frog.jpg&#39;], split = 0.8, dest = &#39;&#39;)

    test.download\_images(maxN = 500)

    test.filter\_imgs()

    #test.multiply\_imgs(test.Query)

    test.pickle\_images(split = 0.9, dest = &#39;test1&#39;)

**&quot;Model&quot; Class:**

Creates a basic keras convolutional neural network and trains it based on the images processed by the &quot;ImageCollector&quot; class. In the future I aim to make the user be able to choose all parameters and size of the network. At the moment it is mostly fixed.

**Methods:**

- **--**** \_\_init\_\_(self,epoch, num\_out, hold\_prob = 0.7, learning\_rate = 0.0001, layers = 2, filters = [32,64, 128], name = &#39;test\_model&#39;)**
- **--**** get\_images(self, dest):** Loads a pickle from dest.
- **--**** next\_batch(self, data, data\_labels,  steps):** Outputs a batch of images and labels from data and data\_labels of size steps. The batch is random.
- **--**** train\_layers(self):** Builds and trains the convolutional neural network. Return the model created.

**Ex:**

    model = Model(80,4, layers = 2)

    model.get\_images(&#39;test1.pickle&#39;)

    model.train\_layers()



**Results:**

**Test 1:**

- **--**** Queries ****=** [&#39;airplane.jpg&#39;, &#39;dog.jpg&#39;, &#39;trees.jpg&#39;, &#39;frog.jpg&#39;]
- **--**** Number of Images =** 500
- **--**** Epoch = 80**
- **--**** Train/Test split =** 9

This resulted in an accuracy of 85%.

**Test 2:**

- **--**** Queries ****=** [&#39;airplane.jpg&#39;, &#39;dog.jpg&#39;, &#39;trees.jpg&#39;, &#39;frog.jpg&#39;, &#39;ship.jpg&#39;, &#39;bird.jpg&#39;]
- **--**** Number of Images =** 500
- **--**** Epoch = 80**
- **--**** Train/Test split =** 9

This resulted in an accuracy of 75%.

The biggest problem is the fact that we don&#39;t have enough quality images to train the network. Not all images downloaded are of exactly what we asked for and the more images we ask for, the lower quality they become.

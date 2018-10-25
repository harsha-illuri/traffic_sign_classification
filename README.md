
## Classification of Belgium traffic sign images using Neural Nets

Uses a neural network with 3 hidden layers

stochastic gradient descent as optimizer

learning rate = 0.001

loss function = categorical_crossentropy


```python
import tensorflow as tf
import sklearn as sk
import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
```

    Using TensorFlow backend.
    

### Data is avaiable at https://btsd.ethz.ch/shareddata/


```python
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    images = np.array(images)
    labels = np.array(labels)
    images28 = [skimage.transform.resize(image, (28, 28)) for image in images]
    images28 = np.array(images28)
    images28 = skimage.color.rgb2gray(images28)
    images28 = np.array(images28)
    return images28, labels

train_data_directory = "Training"
test_data_directory = "Testing"
```


```python
# load trainning data

images, labels = load_data(train_data_directory)
train_x = np.reshape(images, (4575, 784))
train_y = keras.utils.to_categorical(labels, 62)

# load test data
images, labels = load_data(test_data_directory)
test_x = np.reshape(images, (2520, 784))
test_y = keras.utils.to_categorical(labels, 62)
```

    c:\program files\python36\lib\site-packages\skimage\transform\_warps.py:105: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    c:\program files\python36\lib\site-packages\skimage\transform\_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
      warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
    


```python
# define Model
model = Sequential()
model.add(Dense(units = 128, activation="relu", input_shape = (784,)))
model.add(Dense(units = 128, activation="relu"))
model.add(Dense(units = 128, activation="relu"))
model.add(Dense(units=62,activation="softmax"))
model.compile(optimizer=SGD(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
```


```python
# train Model
history = model.fit(train_x,train_y,validation_data = (test_x, test_y), batch_size=32,epochs=300,verbose=2)
```

    Train on 4575 samples, validate on 2520 samples
    Epoch 1/300
     - 3s - loss: 4.1142 - acc: 0.0422 - val_loss: 4.1377 - val_acc: 0.0540
    Epoch 2/300
     - 1s - loss: 4.0439 - acc: 0.0979 - val_loss: 4.0763 - val_acc: 0.0579
    Epoch 3/300
     - 0s - loss: 3.9714 - acc: 0.1023 - val_loss: 4.0103 - val_acc: 0.0325
    .
    .
    .
    .
     - 0s - loss: 0.4177 - acc: 0.8984 - val_loss: 0.6753 - val_acc: 0.8238
    Epoch 297/300
     - 0s - loss: 0.4163 - acc: 0.9010 - val_loss: 0.6669 - val_acc: 0.8246
    Epoch 298/300
     - 1s - loss: 0.4141 - acc: 0.9003 - val_loss: 0.6683 - val_acc: 0.8258
    Epoch 299/300
     - 0s - loss: 0.4127 - acc: 0.9016 - val_loss: 0.6664 - val_acc: 0.8298
    Epoch 300/300
     - 0s - loss: 0.4107 - acc: 0.9008 - val_loss: 0.6535 - val_acc: 0.8298
    


```python
# plot of accuray with iterations
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png](TF_files/TF_7_0.png)



```python
# plot of loss with iterations
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


![png](TF_files/TF_8_0.png)


## Results

After hyperparameter tuning got a classification accuray of 82.9% with this model

Will have to experiment with CNN to improve the results

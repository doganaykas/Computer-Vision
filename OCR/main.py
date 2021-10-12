# Importing the necessary libraries
from __future__ import print_function
from keras.models import Sequential
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import scipy.io
from tensorflow.keras.optimizers import Adam
import argparse
import sys
import string
import numpy as np
import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation
import keras
from keras.models import Sequential
from keras.optimizers import adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model

# Setting random seed for exact results as report
np.random.seed(14)

# Parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-tr', '--train_input', type=str, default='emnist-byclass-train.csv')
parser.add_argument('-te', '--test_input', type=str, default='emnist-byclass-test.csv')
parser.add_argument('-cl', '--classes', type=int, default=62)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr','--learning_rate',type=float, default=0.001)
parser.add_argument('-e','--epochs',type=int, default=10)
parser.add_argument('-im','--images_file',type=str, default='input_images')
parser.add_argument('-o','--output_file',type=str, default='translations.txt')

args = parser.parse_args()
train = pd.read_csv(args.train_input)
test = pd.read_csv(args.test_input)
num_classes=args.classes
learning_rate=args.learning_rate
epochs=args.epochs
batch_size=args.batch_size
images=args.images_file
output=args.output_file

# Preprocessing train data
x_train = train.iloc[:,1:]
x_train = x_train.astype('float32')
x_train /= 255
print ("x_train:",x_train.shape)
x_train = np.asarray(x_train)
x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')
y_train = train.iloc[:,0]
y_train = np_utils.to_categorical(y_train, num_classes)

# Preprocessing test data
x_test = test.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print ("x_test:",x_test.shape)
x_test = np.asarray(x_test)
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32')

y_test = test.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)

# Reshaping the EMNIST dataset to correct orientation
# The first loop converts the images 90 degrees clockwise, second one inverts along them the vertical
x_tr=[]
x_te=[]
for i in range(len(x_train)):
    a = x_train[i].swapaxes(-2,-1)[...,::-1]
    a = np.fliplr(a)
    x_tr.append(a)
x_tr = np.array(x_tr)
x_train = x_tr
for i in range(len(x_test)):
    b = x_test[i].swapaxes(-2,-1)[...,::-1]
    b = np.fliplr(b)
    x_te.append(b)
x_te = np.array(x_te)
x_test = x_te
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# CNN Model
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam(learning_rate=learning_rate),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot confusion matrix 
confusion_matrix = np.zeros((y_train.shape[1],y_train.shape[1]))
result = model.evaluate(x_test, y_test, verbose=2)
_ = result[0]
acc = result[1]
print('test accuracy: ',acc)
y_pred = model.predict(x_test)
for i in range(y_test.shape[0]):
    confusion_matrix[np.argmax(y_pred[i,:]),np.argmax(y_test[i,:])] += 1
plt.figure(figsize=(10,10))
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(62),''.join([str(_) for _ in range(10)]) + string.ascii_uppercase[:26] + string.ascii_lowercase[:26])
plt.yticks(np.arange(62),''.join([str(_) for _ in range(10)]) + string.ascii_uppercase[:26] + string.ascii_lowercase[:26])
plt.savefig('confusion_matrix.png')


# Function to find individual characters from the input images
# Takes a color image
# Returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    #Denoising the input
    image = skimage.restoration.denoise_wavelet(image,multichannel=True, convert2ycbcr=True)
    #Converting to grayscale
    image = skimage.filters.gaussian(image,3.2)
    image = skimage.color.rgb2gray(image)
    
    
    #Threshold
    image = skimage.morphology.erosion(image)
    threshold = skimage.filters.threshold_otsu(image)
    #Morphology: The morphological closing on an image is defined as a dilation followed by an erosion. 
    #Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks. 
    bw = skimage.morphology.closing(image>threshold,skimage.morphology.square(3))
    
    
    bw = (bw == False)
    #Remove artifacts connected to image border
    cleared = skimage.segmentation.clear_border(bw)
    label_image = skimage.measure.label(cleared)
    #skip small boxes
    for region in skimage.measure.regionprops(label_image):
        if region.area > 500:
            minr, minc, maxr, maxc = region.bbox
            minr -= 3
            minc -= 3
            maxr += 3
            maxc += 3
            bboxes.append((minr, minc, maxr, maxc))
    bw = (bw == False)
    return bboxes, bw

# Finding the individual characters from inputs, depicting them
for img in os.listdir(images):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join(images,img)))
    bboxes, bw = findLetters(im1)
    plt.figure()

    plt.imshow(bw, cmap=plt.cm.gray)
    plt.imshow(im1)
    #skimage.io.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    name = 'crop' + str(os.listdir(images).index(img)) + '.png'
    plt.savefig(name)

# Converting the scanned characters to their machine encoded translation
num = 0
res = []
letters = np.array([str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in string.ascii_lowercase[:26]])
for img in os.listdir(images):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join(images,img)))
    bboxes, bw = findLetters(im1)
    num += 1
    minrList = []
    result = ''
    #sort bboxes to make sure it is selected left to right, top to bottom as in image
    start = 0
    count = 0
    bboxes_new = []
    for i in range(len(bboxes)):
        count += 1
        if abs(bboxes[i][0] - bboxes[i+1][0]) > 100:
            bboxes_s = bboxes[start:count]
            bboxes_s.sort(key = lambda x:x[1])
            bboxes_new.extend(bboxes_s)
            start = count
        if i == len(bboxes)-2:
            bboxes_s = bboxes[start:]
            bboxes_s.sort(key = lambda x:x[1])
            bboxes_new.extend(bboxes_s)
            break
        
    a=[]
    for bbox in bboxes_new:
        minr, minc, maxr, maxc = bbox
        minrList.append(minr)
        crop = (bw[minr:(maxr+1),minc:(maxc+1)]).astype(float)
        #pad zero
        crop = np.pad(crop,(int((maxr-minr)/5),int((maxc-minc)/5)),'edge')
        crop = skimage.transform.resize(crop,(28,28,1))
        a.append(crop)
    a = np.array(a)
    a = 1 - a
    probs = model.predict(a)
    for i in range(len(probs)):
        result += letters[np.argmax(probs[i])]
    row = 1
    count = 0
    start = 0
    res_t = []
    for i in range(len(minrList)):
        count += 1
        if i == len(minrList) -1:
            break
        if abs(minrList[i] - minrList[i + 1]) > 100:
            print(result[start:count])
            r = result[start:count]
            res_t.append(r)
            start = count
            row += 1
    print(result[start:count])
    r = result[start:count]
    res_t.append(r)
    print('there are %d rows in %d-th image'%(row,num))
    res.append(res_t)

# Extracting the translations through txt file
with open(output, "w") as output:
    for i in res:
        if res.index(i)!=0:
            output.write('\n')
            
        output.write(str(res.index(i)))
        output.write(':')
        output.write('\n')
        for j in i:            
            output.write(str(j))
            output.write('\n')    
from tensorflow.keras.layers import Dense,Input,Flatten,Lambda
from tensorflow.keras.models import Model
from keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Initializing Image size for standard dimensions
# =============================================================================
Image_size=[224,224]

train_path="F:/ML_project/kaggle/Cotton Disease/train"
val_path="F:/ML_project/kaggle/Cotton Disease/val"


vgg=VGG16( include_top=False,
    weights='imagenet',
    input_shape=Image_size+[3] )

for layers in vgg.layers:
    layers.trainable=False



x=Flatten()(vgg.output)
prediction=Dense(units=4,activation='softmax') (x)

model=Model(inputs=vgg.output,outputs=prediction)

model.summary()

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics='accuracy')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IDG=ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1./255
    )

train_data=IDG.flow_from_directory(train_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32
     )

test_data=IDG.flow_from_directory(val_path,
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32)

model_fit=model.fit( train_data,
    epochs=10,
    validation_data=test_data,
    steps_per_epoch=len(train_data),
    validation_steps=len(test_data))

y_pred=model.predict(test_data)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model.save('model_vgg.h5')

y_pred_class=np.argmax(y_pred,axis=1)

import numpy as np

model.load_model('model_vgg.h5')

test_image=image.load_img("F:/ML_project/kaggle/Cotton Disease/test",target_size=(224,224))

image_array=image.img_to_array(test_image)

image_array=image_array/255

image_array=np.expand_dims(image_array,axis=0)

predict=model.predict(image_array)

predict=np.argmax(predict,axis=1)

print('The Output class for given image is {}'.format(predict))
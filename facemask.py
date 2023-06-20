import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime


# Building model to classify between mask and no mask
input_shape = (150, 150, 3)

inputs = keras.Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(100, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    'train',
    target_size=input_shape[:2],
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=input_shape[:2],
    batch_size=16,
    class_mode='binary'
)

# Training the model
model.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=10,
    validation_data=test_set,
    validation_steps=len(test_set)
)

model.save('mymodel.h5')

# Implementing live detection of face mask
model = keras.models.load_model('mymodel.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img = cap.read()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        test_image = tf.keras.utils.load_img('temp.jpg', target_size=input_shape[:2])
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        pred = model.predict(test_image)[0][0]
        
        if pred == 1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
          
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
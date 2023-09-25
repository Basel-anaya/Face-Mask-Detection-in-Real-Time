# Import necessary libraries
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
import cv2
import datetime

# Define input shape for the CNN model
input_shape = (150, 150, 3)

# Build and compile the CNN model
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

# Data Generators for Training and Testing

# Create an image data generator for training data.
# - `rescale`: Normalize pixel values to be in the [0, 1] range.
# - `shear_range`: Randomly apply shearing transformations.
# - `zoom_range`: Randomly apply zooming transformations.
# - `horizontal_flip`: Randomly flip images horizontally.
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create an image data generator for testing data.
# - `rescale`: Normalize pixel values to be in the [0, 1] range.
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate batches of training data from the 'train' directory.
# - `target_size`: Resize images to the specified dimensions (150x150 pixels).
# - `batch_size`: Number of samples per batch.
# - `class_mode`: Binary classification ('binary' means two classes: mask and no mask).
train_set = train_datagen.flow_from_directory(
    'train',
    target_size=input_shape[:2],
    batch_size=16,
    class_mode='binary'
)

# Generate batches of testing data from the 'test' directory.
# - `target_size`: Resize images to the specified dimensions (150x150 pixels).
# - `batch_size`: Number of samples per batch.
# - `class_mode`: Binary classification ('binary' means two classes: mask and no mask).
test_set = test_datagen.flow_from_directory(
    'test',
    target_size=input_shape[:2],
    batch_size=16,
    class_mode='binary'
)

# Train the Model

# Fit the model using the training data.
# - `steps_per_epoch`: Number of batches of samples to use per epoch.
# - `epochs`: Number of training epochs.
# - `validation_data`: Use the testing data for validation during training.
# - `validation_steps`: Number of batches of testing data to use for validation.
model.fit(
    train_set,
    steps_per_epoch=len(train_set),
    epochs=10,
    validation_data=test_set,
    validation_steps=len(test_set)
)

# Save the trained model
model.save('mymodel.h5')

# Implement real-time face mask detection
model = keras.models.load_model('mymodel.h5')

# Initialize the video capture and face detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    # Read a frame from the video feed
    _, img = cap.read()
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Crop the detected face
        face_img = img[y:y+h, x:x+w]
        
        # Save the cropped face as a temporary image
        cv2.imwrite('temp.jpg', face_img)
        
        # Load and preprocess the temporary image for prediction
        test_image = tf.keras.utils.load_img('temp.jpg', target_size=input_shape[:2])
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make predictions using the loaded model
        pred = model.predict(test_image)[0][0]
        
        # Display the result on the video feed
        if pred == 1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, 'NO MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, 'MASK', ((x+w)//2, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        # Display the current date and time on the video feed
        datet = str(datetime.datetime.now())
        cv2.putText(img, datet, (400, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
          
    # Display the frame with face detection results
    cv2.imshow('img', img)
    
    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
        break
    
cap.release()
cv2.destroyAllWindows()

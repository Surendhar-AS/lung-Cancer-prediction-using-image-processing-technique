import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
train_path = "Lung_cancer/Train/"
validation_path = "Lung_cancer/Test cases/"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 1
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
model = load_model('lung_cancer_model.h5') 
image_path = "C:\\Users\\SURENDHAR\\OneDrive\\Documents\\git\\lung Cancer prediction using image processing technique\\ctc.jpeg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
img = cv2.resize(img, (224, 224))  
img_array = np.expand_dims(img, axis=0)  
img_array = img_array / 255.0 
predictions = model.predict(img_array)
class_labels = ['Benign cases', 'Malignant cases', 'Normal cases']
predicted_class_index = np.argmax(predictions)
predicted_class_label = class_labels[predicted_class_index]
print("Predicted class:", predicted_class_label)
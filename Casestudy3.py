import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
"dataset/train",
target_size=(img_size,img_size),
batch_size=batch_size,
class_mode='categorical',
subset='training'
)

val_generator = train_datagen.flow_from_directory(
"dataset/train",
target_size=(img_size,img_size),
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128,activation='relu'))

model.add(layers.Dense(2,activation='softmax'))

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)

model.fit(train_generator,epochs=10,validation_data=val_generator)

# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
        'E:/Mini Project/cardataset/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all test images
validation_generator = validation_data_gen.flow_from_directory(
        'E:/Mini Project/cardataset/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
veh_model = Sequential()
                # adding convolutional layers
veh_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1))) 
veh_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Dropout(0.25))
                # adding more layers to the model
veh_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
veh_model.add(MaxPooling2D(pool_size=(2, 2)))
veh_model.add(Dropout(0.25))
        
veh_model.add(Flatten())        # flattening
veh_model.add(Dense(1024, activation='relu'))
veh_model.add(Dropout(0.5))
veh_model.add(Dense(4, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

                # compilinng the model
veh_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the neural network/model
veh_model_info = veh_model.fit_generator(
        train_generator,
        steps_per_epoch=11451 // 64,
        epochs=22,
        validation_data=validation_generator,
        validation_steps=3360
         // 64)

# save model structure in jason file
model_json = veh_model.to_json()
with open("veh_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
veh_model.save_weights('vehicle_model.h5')
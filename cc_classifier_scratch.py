from keras import layers
from keras import models

image_width = 224
image_height = 224

def create_model():
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(image_width, image_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    
    return model


from keras import optimizers

model = None
saved_model = None

if saved_model is None:
    model = create_model()
else :
    print("Loading model %s" % saved_model)
    model = load_model(saved_model)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
model.summary()



import os
# training data generator + image augmentation
from keras.preprocessing.image import ImageDataGenerator

base_dir = './data/clamps'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# validation data should not be augmented
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_width, image_height),
    batch_size=64,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_width, image_height),
    batch_size=64,
    class_mode='categorical')


from keras.callbacks import TensorBoard,ModelCheckpoint

tensorboard = TensorBoard(log_dir='./logs')

checkpointer = ModelCheckpoint(
    filepath='./checkpoints/' +'scratch' + \
        '.{epoch:03d}-{val_loss:.3f}.hdf5',
    verbose=1,
    save_best_only=True)

callbacks = [tensorboard,checkpointer]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=220,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=75,
    callbacks=callbacks)
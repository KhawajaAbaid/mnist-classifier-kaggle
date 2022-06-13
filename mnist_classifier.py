import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# fetching data and some preprocessing
train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test_images = test_data.values
test_images = test_images.reshape(28000, 28, 28)

train_labels = train_data['label'].values
del train_data['label']
train_images = train_data.values
train_images = train_images.reshape(42000, 28, 28)

# Building the model
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Rescaling(1. / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, use_bias=False)(x)

for filters in [32, 64, 128, 256, 512]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPool2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(filters, 1, strides=2, padding="same")(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
callbaccs = [
    keras.callbacks.ModelCheckpoint(
    filepath="fr_fr_100epoch_my_proud_xception_the_best.keras",
    save_best_only = True,
    monitor="val_accuracy")
]

# beginning training
model.fit(train_images, train_labels, epochs=100,
          batch_size=64,
         callbacks = callbaccs, validation_split=0.1)

# getting the best model and making predictions
best_model = keras.models.load_model('fr_fr_100epoch_my_proud_xception_the_best.keras')
predx = best_model.predict(test_images)
predx_fr = [np.argmax(row) for row in predx]

# storing data for output
idx = test_data.index.values
idx += 1
data = np.asarray([idx, predx_fr])
data = data.T
my_submission = pd.DataFrame(data=data, columns=['ImageId', 'Label'], index=None)
my_submission.to_csv('submission.csv', index=False)
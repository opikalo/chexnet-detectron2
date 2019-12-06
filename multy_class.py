import os

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from sklearn.preprocessing import MultiLabelBinarizer



mlb = MultiLabelBinarizer()

ROOT_DATA = '/media/oleksiyp/easystore/data/chestx'

DATA_ENRY = os.path.join(ROOT_DATA, 'Data_Entry_2017.csv')

#DATA_IMAGES = os.path.join(ROOT_DATA, 'images')
DATA_IMAGES = os.path.join('/home/oleksiyp/Documents/images')

train = pd.read_csv(DATA_ENRY)

train['Finding Labels'] = train['Finding Labels'].str.split("|")

# drop 'No Finding' label
def purge_value(x, value):
    if value in x:
        x.remove(value)

    return x

train['Finding Labels'].apply(lambda x: purge_value(x, 'No Finding'))

mlb = MultiLabelBinarizer()
mlb.fit(train['Finding Labels'])

train['Finding Labels'] = train['Finding Labels'].apply(lambda x: mlb.transform([x])[0])


train_image = []
max_upper_range = train.shape[0]
upper_range = 1000
for i in tqdm(range(upper_range)):
    image_path = os.path.join(DATA_IMAGES, train['Image Index'][i])
    img = image.load_img(image_path, target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

X = np.array(train_image)
y = np.vstack(train['Finding Labels'][:upper_range])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='sigmoid'))

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

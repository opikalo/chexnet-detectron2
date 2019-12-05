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

DATA_ENRY = '/media/oleksiyp/easystore/data/chestx/Data_Entry_2017.csv'

train = pd.read_csv(DATA_ENRY)

train['Finding Labels'] = train['Finding Labels'].str.split("|")

mlb = MultiLabelBinarizer()
mlb.fit_transform(train['Finding Labels'])

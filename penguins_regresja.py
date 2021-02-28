import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import TensorBoard

# Wczytanie pliku
colnames = ['species', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']

dataframe = pd.read_csv('penguins_cleaned.csv', sep=',', header=None, names=colnames)

X = dataframe.iloc[:, 1:]
y = dataframe.iloc[:, 0]
# print(y)
X = pd.get_dummies(X)
encoder = LabelEncoder()
label = encoder.fit_transform(y)
y = pd.get_dummies(label).values
print(y)
print(X)
print(X.columns)

train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.28, random_state=42)

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))

# Model regresji liniowej

linear_model = keras.Sequential([
    normalizer,
    keras.layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mse', metrics=['mse'])
name = "linear_regression"
tensorboard = TensorBoard(log_dir="logs/{}".format(name))
linear_model.fit(train_features, train_labels, epochs=100,
                                  verbose=3, validation_split=0.21, callbacks=[tensorboard])
#tensorboard --logdir=logs/

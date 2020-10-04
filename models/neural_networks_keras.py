import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.layers.core import Dropout

def build_model():
    # Neural networks model building with Keras and TensorFlow
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=110)
    model = Sequential()
    input_shape = X_train.shape[1]
    model.add(layers.Dense(1024, input_shape=(input_shape,), activation="relu"))

    # Dropout for avoiding overfitting
    model.add(Dropout(0.2))
    model.add(layers.Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    fit_keras = model.fit(X_train, y_train, epochs=100, verbose=True, validation_data=(X_test, y_test), batch_size=30)
    return model

def model_performance(model):
    accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training score: {:.5f}".format(accuracy[0]))
    print("Training accuracy: {:.5f}\n".format(accuracy[1]))

    accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing score: {:.5f}".format(accuracy[0]))
    print("Testing accuracy: {:.5f}\n".format(accuracy[1]))

if __name__ == '__main__':
    model = build_model()
    model_performance(model)

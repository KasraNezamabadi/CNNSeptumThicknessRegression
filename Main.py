import numpy as np
import cv2
import DataSource as ds
import CNN
import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


image_width = 96
image_height = 96




def train_model(batch_size = 40, nb_epoch = 20):

    # X_train, y_train, X_valid, y_valid, X_test, y_test = ds.get_dataset()
    #
    #
    # model = CNN.get_model(image_width=image_width, image_height=image_height)
    # model.fit(X_train, y_train,
    #           batch_size=batch_size,
    #           nb_epoch=nb_epoch,
    #           verbose=1,
    #           validation_data=(X_valid, y_valid))

    X_train, y_train, X_test, y_test = ds.get_dataset2()

    model2 = CNN.get_model2(image_width=image_width, image_height=image_height)
    model2.fit(x=X_train, y=y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1)

    predictions_valid = model2.predict(X_test, batch_size=20, verbose=1)

    print('Done')
    predictions_valid = predictions_valid.flatten()
    compare = pd.DataFrame(data={'Original': y_test, 'Prediction': predictions_valid})

    compare.to_csv('compare.csv')

    return model2

train_model()



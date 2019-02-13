from keras import backend as KerasBackend
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.layers import Conv2D, Input
from keras.models import Model

KerasBackend.set_image_data_format('channels_last')  # TF dimension ordering in this code
smooth = 1.


def get_model2(image_width, image_height):

    inputs = Input((image_width, image_height, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='valid')(inputs)
    conv1 = Dropout(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='valid')(pool1)
    conv2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    flat = Flatten()(pool2)

    dens1 = Dense(1024, activation='sigmoid')(flat)
    dens1 = Dropout(0.5)(dens1)

    dens2 = Dense(512, activation='sigmoid')(dens1)
    dens2 = Dropout(0.5)(dens2)

    out = Dense(1, activation='linear')(dens2)

    model = Model(inputs=[inputs], outputs=[out])
    model.compile(optimizer=Adadelta(), loss='mean_squared_error')

    return model
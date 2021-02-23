########################################################################
# import python-library
########################################################################
# from import
import keras.models
from keras import backend as K
from keras.layers import Input, Concatenate
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model


########################################################################
# keras model
########################################################################
def get_model(n_frames, n_mels, n_conditions, lr):
    """
    define the keras model
    the model based on MobileNetV2
    """

    sub_model = MobileNetV2(input_shape=(n_frames, n_mels, 3),
                            alpha=0.5, 
                            weights=None,
                            classes=n_conditions)
   
    x = Input(shape=(n_frames, n_mels, 1))
    h = x
    h = Concatenate()([h, h, h])
    h = sub_model(h)

    model = Model(x, h)

    model.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

#########################################################################

def load_model(file_path):
    return keras.models.load_model(file_path, compile=False)

def clear_session():
    K.clear_session()
    
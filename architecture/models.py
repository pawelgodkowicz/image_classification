from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16

class Vgg16():

    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)

        top_model = Sequential()
        top_model.add(Flatten())
        top_model.add(Dense(units=512, activation='relu'))
        top_model.add(Dropout(0.3))
        top_model.add(Dense(units=self.num_classes, activation=self.final_activation))

        transfer_model = Sequential()
        transfer_model.add(base_model)
        transfer_model.add(top_model)
        transfer_model.layers[0].trainable = False
        transfer_model.summary()

        return transfer_model
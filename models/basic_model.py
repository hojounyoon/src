from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import RMSprop, Adam

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here
        # you have to initialize self.model to a keras model
        self.model = Sequential()
        self.model.add(layers.Input(shape=input_shape))
        self.model.add(layers.Rescaling(1./255))
        
        self.model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size = (2, 2)))

        self.model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size = (2, 2)))

        self.model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size = (2, 2)))
        
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dropout(0.5))
        
        self.model.add(layers.Dense(units = categories_count, activation='softmax'))
    
    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

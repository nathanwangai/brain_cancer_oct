import tensorflow as tf

'''
See Figure 2 of the manuscript for a diagram of the ensemble architecture (https://doi.org/10.1364/BOE.477311)

----- Contents of this file -----
B_frame_CNN: convolutional neural network for B-frame slices
texture_CNN: convolutional neural network for texture slices
ensemble_MLP: multi-layer perceptron for concatenated B_frame_CNN and texture_CNN embeddings

----- Methods -----
- __init__(): initialize network layers (requried by sub-classed TensorFlow models)
- call(): define how the network layers are connected (requried by sub-classed TensorFlow models)
- model(): builds the network graph, which is required before extracting intermediate activations
- get_embedding() and get_gradcam(): returns the activation of intermediate network layers (embeddings)
'''

# ====================================================================================================

class B_frame_CNN(tf.keras.Model):
    def __init__(self, filter_size, activation, padding):
        super().__init__()
        # convolutional layers
        self.c1 = tf.keras.layers.Conv2D(32, filter_size, strides=(2,1), activation=activation, padding=padding)
        self.c2 = tf.keras.layers.Conv2D(64, filter_size, strides=2, activation=activation, padding=padding)
        self.p1 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
        self.c3 = tf.keras.layers.Conv2D(128, filter_size, strides=2, activation=activation, padding=padding, name="gradmaps") # get_gradcam() returns the activation at this layer
        self.p2 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
        self.c4 = tf.keras.layers.Conv2D(256, filter_size, strides=2, activation=activation, padding=padding)
        self.c5 = tf.keras.layers.Conv2D(256, filter_size, strides=2, activation=activation, padding=padding)

        # dense layers
        self.f1 = tf.keras.layers.Flatten()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.d1 = tf.keras.layers.Dense(64, activation=None, name="embedding") # get_embedding() returns the activation at this layer
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.d2 = tf.keras.layers.Dense(2, activation='softmax')
        self.full_model = None

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.p1(x)
        x = self.c3(x)
        x = self.p2(x)
        x = self.c4(x)
        x = self.c5(x)
        
        x = self.f1(x)
        x = self.dropout1(x, training=True)
        x = self.d1(x)
        x = self.dropout2(x, training=True)
        return self.d2(x)
        
    def model(self): # calling self.model() builds the network graph, allowing us to access intermediate layers
        inputs = tf.keras.Input(shape=(200,100,1))
        self.full_model = tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
        return self.full_model

    def get_embedding(self):
        intermediate = tf.keras.models.Model(inputs=self.model().input, outputs=self.full_model.get_layer("embedding").output)
        return intermediate

    def get_gradcam(self): 
        intermediate2 = tf.keras.models.Model(inputs=self.model().input, outputs=[self.full_model.get_layer("gradmaps").output, self.full_model.output])
        return intermediate2

# ====================================================================================================

class texture_CNN(tf.keras.Model):
    def __init__(self, filter_size, activation, padding):
        super().__init__()
        # convolutional layers
        self.c1 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding)
        self.p1 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
        self.c2 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding, name="gradmaps") # get_gradcam() returns the activation at this layer
        self.p2 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
        self.c3 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding)
        self.c4 = tf.keras.layers.Conv2D(16, filter_size, strides=2, activation=None, padding=padding)

        # dense layers
        self.f1 = tf.keras.layers.Flatten(name="embedding") # get_embedding() returns the activation at this layer
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.d1 = tf.keras.layers.Dense(2, activation='softmax')
        self.full_model = None

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.p1(x)
        x = self.c2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.c4(x)
        
        x = self.f1(x)
        x = self.dropout1(x, training=True)
        return self.d1(x)
        
    def model(self): # calling self.model() builds the network graph, allowing us to access intermediate layers
        inputs = tf.keras.Input(shape=(100,100,1))
        self.full_model = tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
        return self.full_model

    def get_embedding(self):
        intermediate = tf.keras.models.Model(inputs=self.model().input, outputs=self.full_model.get_layer("embedding").output)
        return intermediate

    def get_gradcam(self): 
        intermediate2 = tf.keras.models.Model(inputs=self.model().input, outputs=[self.full_model.get_layer("gradmaps").output, self.full_model.output])
        return intermediate2
    
# ====================================================================================================

class ensemble_MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = tf.keras.layers.Dense(64, activation=None)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.result = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.hidden(inputs)
        x = self.dropout1(x, training=True) # keeps dropout active during inference
        return self.result(x)

    def model(self):
        inputs = tf.keras.Input(shape=(1, 128))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
import numpy as np
import tensorflow as tf
import bframe_processing as bfp

class BFrameCNN(tf.keras.Model):
	def __init__(self, filter_size, activation, padding):
		super().__init__()
		self.c1 = tf.keras.layers.Conv2D(32, filter_size, strides=(2,1), activation=activation, padding=padding)
		self.c2 = tf.keras.layers.Conv2D(64, filter_size, strides=2, activation=activation, padding=padding)
		self.p1 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
		self.c3 = tf.keras.layers.Conv2D(128, filter_size, strides=2, activation=activation, padding=padding, name="gradmaps")
		self.p2 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
		self.c4 = tf.keras.layers.Conv2D(256, filter_size, strides=2, activation=activation, padding=padding)
		self.c5 = tf.keras.layers.Conv2D(256, filter_size, strides=2, activation=activation, padding=padding)

		self.f1 = tf.keras.layers.Flatten()
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		self.d1 = tf.keras.layers.Dense(64, activation=None, name="embedding")
		self.dropout2 = tf.keras.layers.Dropout(0.5)
		self.d2 = tf.keras.layers.Dense(2, activation='softmax')

		inputs = tf.keras.Input(shape=(200,100,1))
		self.full_model = tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

	def call(self, inputs):
		x = self.c1(inputs)
		x = self.c2(x)
		x = self.p1(x)
		x = self.c3(x)
		x = self.p2(x)
		x = self.c4(x)
		x = self.c5(x)

		x = self.f1(x)
		x = self.dropout1(x)
		x = self.d1(x)
		x = self.dropout2(x)
		return self.d2(x)

	def get_intermediate(self, layer_name):
		intermediate = tf.keras.models.Model(inputs=self.full_model.input, outputs=self.full_model.get_layer(layer_name).output)
		return intermediate

class TextureCNN(tf.keras.Model):
	def __init__(self, filter_size, activation, padding):
		super().__init__()
		self.c1 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding)
		self.p1 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
		self.c2 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding, name="gradmaps")
		self.p2 = tf.keras.layers.MaxPooling2D(filter_size, strides=2)
		self.c3 = tf.keras.layers.Conv2D(32, filter_size, strides=2, activation=activation, padding=padding)
		self.c4 = tf.keras.layers.Conv2D(16, filter_size, strides=2, activation=None, padding=padding)

		self.f1 = tf.keras.layers.Flatten(name="embedding")
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		self.d1 = tf.keras.layers.Dense(2, activation='softmax')
		self.full_model = None

		inputs = tf.keras.Input(shape=(100,100,1))
		self.full_model = tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

	def call(self, inputs):
		x = self.c1(inputs)
		x = self.p1(x)
		x = self.c2(x)
		x = self.p2(x)
		x = self.c3(x)
		x = self.c4(x)

		x = self.f1(x)
		x = self.dropout1(x)
		return self.d1(x)
			
	def get_intermediate(self, layer_name):
		intermediate = tf.keras.models.Model(inputs=self.full_model.input, outputs=self.full_model.get_layer(layer_name).output)
		return intermediate

class EnsembleMLP(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.hidden = tf.keras.layers.Dense(64, activation=None)
		self.dropout1 = tf.keras.layers.Dropout(0.5)
		self.result = tf.keras.layers.Dense(2, activation='softmax')

		inputs = tf.keras.Input(shape=(1,128))
		self.full_model = tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

	def call(self, inputs):
		x = self.hidden(inputs)
		x = self.dropout1(x)
		return self.result(x)

# load all models
bframe_model = tf.keras.models.load_model("D:\\brain_cancer_oct\saved_models\CNN_bframe")
bframe_cnn = BFrameCNN(3, 'relu', 'same')
bframe_cnn.set_weights(bframe_model.get_weights())
bframe_embedding_cnn = bframe_cnn.get_intermediate("embedding")

texture_model = tf.keras.models.load_model("D:\\brain_cancer_oct\saved_models\CNN_texture")
texture_cnn = TextureCNN(3, 'relu', 'same')
texture_cnn.set_weights(texture_model.get_weights())
texture_embedding_cnn = texture_cnn.get_intermediate("embedding")

ensemble_model = tf.keras.models.load_model("D:\\brain_cancer_oct\saved_models\ensemble_MLP")
ensemble_mlp = EnsembleMLP()
ensemble_mlp.set_weights(ensemble_model.get_weights())

def get_bframe_predictions(bframe_slices):
	return bframe_cnn(bframe_slices)[:,0]

def get_texture_predictions(textures):
	return texture_cnn(textures)[:,0]

def concatenate_embeddings(bframe_slice):
    slice_texture = bfp.normalize(bfp.convert_to_texture(bframe_slice))
    bframe_embedding = bframe_embedding_cnn(np.reshape(bframe_slice, (1,200, 100,1)))
    texture_embedding = texture_embedding_cnn(np.reshape(slice_texture, (1,100,100,1)))

    return np.concatenate((bframe_embedding, texture_embedding), axis=1)

def get_ensemble_predictions(embeddings):
	return ensemble_mlp(embeddings)[:,0]

if __name__ == "__main__":
	pass # do debugging here

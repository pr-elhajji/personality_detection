import tensorflow as tf 
from tensorflow import keras

import tensorflow_addons as tfa 
import tensorflow_hub as hub


vgg16 = keras.applications.VGG16(include_top=False, weights='imagenet', pooling='max')
vgg16.trainable = False


def make_visual_model(name='model'):
    inputs = keras.layers.Input(shape=(10,224,224,3), name='Input')

    # First model 
    x      = keras.layers.TimeDistributed(keras.layers.Rescaling(scale=1./255.0), name='Rescaling')(inputs)
    x      = keras.layers.TimeDistributed(vgg16, name='vgg16')(x)
    x      = keras.layers.LSTM(units=128, return_sequences=True)(x)
    x      = keras.layers.LSTM(units=64)(x)
    x      = keras.layers.Dropout(0.2)(x)

    x      = keras.layers.Dense(units=1024)(x)
    x      = keras.layers.Dense(units=512, activation='relu')(x)


    # Second model 
    y      = keras.layers.TimeDistributed(keras.layers.Rescaling(scale=1./127.5, offset=-1))(inputs)

    vit_extractor = hub.KerasLayer("https://www.kaggle.com/models/spsayakpaul/vision-transformer/frameworks/TensorFlow2/variations/vit-b16-fe/versions/1/", trainable=False) # ../../vit_b16_fe/
    vit_extractor.compute_output_shape = lambda x: (x[0], 768)

    y      = keras.layers.TimeDistributed(vit_extractor)(y)
    y      = keras.layers.LSTM(units=128, return_sequences=True)(y)
    y      = keras.layers.LSTM(units=64)(y)
    y      = keras.layers.Dropout(0.2)(y)
    y      = keras.layers.Dense(units=1024)(y)
    y      = keras.layers.Dense(units=512, activation='relu')(y)


    # Averaget two models
    z      = keras.layers.Average()([x,y])

    z      = keras.layers.Dense(256, activation='relu')(z)
    z      = keras.layers.Dropout(0.5)(z)

    z      = keras.layers.Dense(5, activation='sigmoid')(z)

    return  keras.models.Model(inputs=inputs, outputs=z, name=name)
    # model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
    # keras.utils.plot_model(model, show_shapes=True)

def make_audio_model():
  
  inputs = keras.layers.Input(shape=(15,128))

  x = keras.layers.Conv1D(32, 2)(inputs)
  x = keras.layers.Dropout(0.3)(x)
  x = keras.layers.Conv1D(64, 2)(x)
  x = keras.layers.Dropout(0.3)(x)

  x = keras.layers.LSTM(512, return_sequences=True)(x)
  x = keras.layers.LSTM(256)(x)

  x = keras.layers.Dense(256)(x)
  x = keras.layers.Dropout(0.3)(x)


  x = keras.layers.Dense(5, activation='sigmoid')(x)

  return keras.models.Model(inputs=inputs, outputs=x, name='audio_model')

import numpy as np 
def make_text_glove_model():
   embed_matrix = np.load('../text/embed_matrix.npy')
   vocab_size   = 11050
   sentlen      = 50

   vocab_size = 11052
  
   inputs = keras.layers.Input(shape=(sentlen))
   embed  = keras.layers.Embedding(input_dim=vocab_size, output_dim=100, embeddings_initializer=keras.initializers.Constant(embed_matrix),input_length=sentlen, trainable=False)(inputs)

   x = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(embed)
   x = keras.layers.Conv1D(filters=8, kernel_size=3, activation='relu')(x)
   x = keras.layers.Flatten()(x)
   x = keras.layers.Dense(50, activation='relu')(x)

   y = keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(embed)
   y = keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu')(y)
   y = keras.layers.Flatten()(y)
   y = keras.layers.Dense(50, activation='relu')(y)

   z = keras.layers.Concatenate()([x,y])

   z = keras.layers.Dense(256, activation='relu')(z)
   z = keras.layers.Dense(5, activation='sigmoid')(z)

   return keras.models.Model(inputs=inputs, outputs=z, name='text_model')

  
def load_scene_model(weights=True):
   
   scene_model = make_visual_model(name='scene_model')
   scene_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
   
   if weights:
      scene_model.load_weights('./weights/scene/0225_123154/scene.t5')
   return scene_model

def load_face_model(weights=True):
   face_model = make_visual_model(name='face_model')
   face_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
   
   if weights:
      face_model.load_weights('./weights/face/0225_154249/face.t5')
   return face_model

def load_audio_model(weights=True):
   audio_model = make_audio_model()
   audio_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
   
   if weights:
      audio_model.load_weights('./weights/audio/0225_191323_9005/audio.t5')
   return audio_model

def load_text_glove_model(weights=True):
   text_model = make_text_glove_model()
   text_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(learning_rate=1e-5), metrics=['mae'])
   if weights:
      text_model.load_weights('./weights/text/0226_072643/text.t5')
   return text_model

# Load early fusion 
def load_ef_model(weights=True):
   scene_model = load_scene_model(weights)
   face_model  = load_face_model(weights)
   audio_model = load_audio_model(weights)
   text_model  = load_text_glove_model(weights)

   scene_inputs = keras.layers.Input(shape=(10,224,224,3), name='Scene_input')
   face_inputs  = keras.layers.Input(shape=(10,224,224,3), name='Face_input')
   audio_inputs = keras.layers.Input(shape=(15,128), name='Audio_input')
   text_inputs  = keras.layers.Input(shape=(50), name='Text_input')

   x = scene_model(scene_inputs)
   y = face_model(face_inputs)
   z = audio_model(audio_inputs)
   w = text_model(text_inputs)

   outputs = keras.layers.Average()([x,y,z, w])
   ef_model = keras.models.Model(inputs=[scene_inputs, face_inputs, audio_inputs, text_inputs], outputs=outputs, name='ef_model')
   ef_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
   return ef_model


# Load model fusion 
def load_mf_model():
   scene_model = load_scene_model()
   face_model  = load_face_model()
   audio_model = load_audio_model()
   text_model  = load_text_glove_model()

   scene_inputs = keras.layers.Input(shape=(10,224,224,3), name='Scene_input')
   face_inputs  = keras.layers.Input(shape=(10,224,224,3), name='Face_input')
   audio_inputs = keras.layers.Input(shape=(15,128), name='Audio_input')
   text_inputs  = keras.layers.Input(shape=(50), name='Text_input')

   x = scene_model(scene_inputs)
   y = face_model(face_inputs)
   z = audio_model(audio_inputs)
   w = text_model(text_inputs)

   w = keras.layers.Concatenate(name='Concatenate')([x,y,z,w])
   w = keras.layers.Flatten(name='Flatten')(w)
   w = keras.layers.Dense(100, activation='relu', name='fc1')(w)
   w = keras.layers.Dropout(0.5, name='Dropout')(w)

   outputs = keras.layers.Dense(5, activation='sigmoid', name='prediction')(w)

   mf_model = keras.models.Model(inputs=[scene_inputs, face_inputs, audio_inputs, text_inputs], outputs=outputs, name="mf_model")
   mf_model.compile(loss='mse', optimizer=tfa.optimizers.RectifiedAdam(), metrics=['mae'])
   return mf_model

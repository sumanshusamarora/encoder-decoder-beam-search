from __future__ import print_function, division
import os, sys
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

BATCH_SIZE = 32  # Batch size for training.
EPOCHS = 10  # Number of epochs to train for.
LATENT_DIM = 512  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 50000  # Number of samples to train on.
MAX_NUM_WORDS = 30000
EMBEDDING_DIM = 100

# Where we will store the data
input_texts = [] # sentence in original language
target_texts = [] # sentence in target language
target_texts_inputs = [] # sentence in target language offset by 1


class EncodeDecoderNetwork:
    
    def __init__(self, max_input_seq_len, max_output_seq_len, input_vocab_size, output_vocab_size, 
                 sos_token, eos_token, rnn_dimensions=512, bidirectional=False,
                 encoder_embedding_dim=100, decoder_embedding_dim=100, pad_token=0, lstm=False,
                 embedding_weights = None, embedding_trainable = False, mask_zero=True):
        #Hyperparamers
        self.max_input_seq_len = max_input_seq_len
        self.max_output_seq_len = max_output_seq_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.encoder_embedding_dim = encoder_embedding_dim
        self.decoder_embedding_dim = decoder_embedding_dim
        self.embedding_weights = embedding_weights
        self.embedding_trainable = embedding_trainable
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.encoder_rnn_size = rnn_dimensions
        self.decoder_rnn_size = rnn_dimensions
        self.mask_zero = mask_zero
        self.lstm = lstm
        
            
        ##### Layers #####
        
        #Input layers
        self.encoder_inp = Input(self.max_input_seq_len, name='encoder_inp')
        self.decoder_inp = Input((None,), name='decoder_input')
        
        #Embedding layers
        if self.embedding_weights is None:
            self.encoder_embedding_layer = Embedding(input_dim=self.input_vocab_size, 
                                                     output_dim=self.encoder_embedding_dim, 
                                                     mask_zero=self.mask_zero,
                                                     name='encoder_embedding_layer')
        else:
            self.encoder_embedding_layer = Embedding(input_dim=self.input_vocab_size, 
                                                     output_dim=self.encoder_embedding_dim,
                                                     weights=[self.embedding_weights],
                                                     mask_zero=self.mask_zero,
                                                     trainable=self.embedding_trainable,
                                                     name='encoder_embedding_layer')
            
        self.decoder_embedding_layer = Embedding(input_dim=self.output_vocab_size, 
                                                 output_dim=self.decoder_embedding_dim, 
                                                 name="decoder_embedding_layer")
        #RNN Layers    
        if self.lstm:
            self.encoder_rnn = LSTM(self.encoder_rnn_size, return_sequences=True, return_state=True, name='encoder_lstm')
            self.decoder_rnn = LSTM(self.decoder_rnn_size, return_sequences=True, return_state=True, name="decoder_lstm")
        else:
            self.encoder_rnn = GRU(self.encoder_rnn_size, return_sequences=True, return_state=True, name='encoder_gru')
            self.decoder_rnn = GRU(self.decoder_rnn_size, return_sequences=True, return_state=True, name="decoder_gru")
            
        
        #Dense Layer
        self.decoder_dense = Dense(self.output_vocab_size, activation="softmax", name="decoder_dense")
                  
        self.encoder_model = self.encoder()
        
    def get_decoder_initial_return_states(self, encoder_input, predict=False):
        if not predict:
            internal_state = self.encoder_model(encoder_input)
        else:
            internal_state = self.encoder_model.predict(encoder_input)
            if isinstance(internal_state, list):
                internal_state = [tf.convert_to_tensor(state) for state in internal_state]
            else:
                internal_state = tf.convert_to_tensor(internal_state)
                                                       
            return internal_state
        
    def encoder(self):
        encoder_embedding_out = self.encoder_embedding_layer(self.encoder_inp)
            
        if self.lstm:
            encoder_out, state_h, state_c = self.encoder_rnn(encoder_embedding_out)
            encoder_internal_states = [state_h, state_c]
        else:
            encoder_out, encoder_internal_states = self.encoder_rnn(encoder_embedding_out)
    
        return Model(self.encoder_inp, encoder_internal_states)
    
    
    def decoder(self, inference=False):
        encoder_emb_out = self.encoder_embedding_layer(self.encoder_inp)
        
        if self.lstm:
            encoder_rnn_out, state_h, state_c = self.encoder_rnn(encoder_emb_out)
            encoder_internal_states = [state_h, state_c]
        else:
            encoder_out, encoder_internal_states = self.encoder_rnn(encoder_emb_out)
        
        decoder_embedding = self.decoder_embedding_layer(self.decoder_inp)
        
        if self.lstm:
            decoder_rnn_out, _, _ = self.decoder_rnn(decoder_embedding, initial_state=encoder_internal_states)
        else:
            decoder_rnn_out, _ = self.decoder_rnn(decoder_embedding, initial_state=encoder_internal_states)
            
        
        decoder_dense_out = self.decoder_dense(decoder_rnn_out)
        
        return Model([self.encoder_inp, self.decoder_inp], decoder_dense_out)
    
    def decoder_inference(self):
        internal_state_h = Input(shape=(self.decoder_rnn_size,))
        internal_state_s = Input(shape=(self.decoder_rnn_size,))
        
        if self.lstm:
            internal_state = [internal_state_h, internal_state_s]
        else:
            internal_state = internal_state_h
            
        decoder_input = Input((1,))
        decoder_emb_out = self.decoder_embedding_layer(decoder_input)
        
        
        if self.lstm:
            decoder_rnn_out, state_h, state_c = self.decoder_rnn(decoder_emb_out, initial_state=internal_state)
            new_internal_state = [state_h, state_c]
        else:
            decoder_rnn_out, new_internal_state = self.decoder_rnn(decoder_emb_out, initial_state=internal_state)

        decoder_dense_out = self.decoder_dense(decoder_rnn_out)
        return Model([internal_state, decoder_input], [decoder_dense_out, new_internal_state])
    
    
    def infer(self, x, max_output_len=10):
        infer_batch_size = x.shape[0]
        decoder_initial_state = network.get_decoder_initial_return_states(x, predict=True)
        docoder_input = np.full((infer_batch_size,1), self.sos_token)
        all_sentences =  []
        for i in range(infer_batch_size):
            if self.lstm:
                this_initial_state = [tf.expand_dims(decoder_initial_state[0][i], axis=0), 
                                      tf.expand_dims(decoder_initial_state[1][i], axis=0)]
            else:
                this_initial_state = tf.expand_dims(decoder_initial_state[i], axis=0)
                
            next_docoder_input = tf.expand_dims(docoder_input[i], axis=0)
            this_sentences =  []
            for k in range(max_output_len):
                this_embedding = self.decoder_embedding_layer(next_docoder_input)
                if self.lstm:
                    next_docoder_input, state_h, state_c = self.decoder_rnn(this_embedding, initial_state=this_initial_state)
                    this_initial_state = [tf.convert_to_tensor(state_h), tf.convert_to_tensor(state_c)]
                else:
                    next_docoder_input, this_initial_state = self.decoder_rnn(this_embedding, initial_state=this_initial_state)
                    this_initial_state = tf.convert_to_tensor(this_initial_state)
                next_docoder_input = self.decoder_dense(next_docoder_input)
                next_docoder_input = np.argmax(next_docoder_input)
                if next_docoder_input == self.eos_token:
                    break
                else:
                    this_sentences.append(str(next_docoder_input))
                    next_docoder_input = np.full((x.shape[0],1), next_docoder_input)
                    
            all_sentences.append(" ".join(this_sentences))
        return all_sentences
        
network = EncodeDecoderNetwork(max_input_seq_len=max_len_input, 
                               max_output_seq_len=max_len_target, 
                               input_vocab_size=max(word2idx_inputs.values())+1,
                               output_vocab_size=max(word2idx_outputs.values())+1, 
                               sos_token=word2idx_outputs["<sos>"],  
                               eos_token=word2idx_outputs["<eos>"], 
                               encoder_embedding_dim=EMBEDDING_DIM,
                               decoder_embedding_dim=EMBEDDING_DIM,
                               rnn_dimensions=LATENT_DIM,
                               lstm=True,
                               embedding_trainable=False,
                               embedding_weights=embedding_matrix)
                               
decoder = network.decoder()

from tensorflow.keras.utils import plot_model
plot_model(decoder, show_shapes=True, show_layer_names=True)

decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

r = decoder.fit(
  [encoder_inputs, decoder_inputs], decoder_targets_one_hot,
  batch_size=24,
  epochs=EPOCHS,
  validation_split=0.2,
)

#Random translation generator

idx2word_inputs = {v:k for k,v in word2idx_inputs.items()}
idx2word_outputs = {v:k for k,v in word2idx_outputs.items()}
def translate(x, tokenize=False):
    if tokenize:
        x = tokenizer_inputs.texts_to_sequences([x])
        x = pad_sequences(x, maxlen=max_len_input)
    out=network.infer(x)
    print(f"Original Sentence: {[idx2word_inputs[word] for word in x[0] if word != 0]}")
    print(f"Translation: {[idx2word_outputs[int(word)] for word in out[0].split() if word != 0]} \n.........................")


while True:
    i = np.random.choice([num for num in range(encoder_inputs.shape[0]) if num < encoder_inputs.shape[0]-1])
    x = encoder_inputs[i:i+1]
    translate(x)
    sentence = None
    response = input("Do you want to test another translation? (y/n)")
    if not response.lower() == 'y':
        break
    

#Take sentence from user and translate
response = input("Enter english sentence.")
translate(response, tokenize=True)

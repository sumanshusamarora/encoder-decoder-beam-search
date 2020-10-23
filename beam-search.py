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

class BeamSearch:
    
    def __init__(self, network, beam_width=3, max_output_seq_len=10):
        self.beam_width = beam_width
        self.max_output_seq_len = max_output_seq_len
        self.network = network
        self.decoder = self.network.decoder_inference()
        self.flatten = Flatten()
        self.final_words = {k:[] for k in range(self.beam_width)}
        self.final_probs = {k:[] for k in range(self.beam_width)}
        self.final_internal_states = {k:None for k in range(self.beam_width)}
        self.final_sequence = []
        
    def squeeze_dimentions(self, arr):
        product = 1
        arr_shape = arr.shape
        for sh in arr_shape:
            product = product*sh
        return K.reshape(arr, (product,))
          
        
    def calculate_combined_probability(self, old_proba, new_probas=None, alpha=1):
        """
        Example - Given t1, t2, .. tn proba
        :params:
        :probabilities Probabilities of all 
        """
        #(K.sum(-1*K.log(probabilities)))/(len(probabilities)**alpha)
        try:
            number_of_elements = len(old_proba)+1
        except:
            number_of_elements = max(old_proba.shape)
        
        if new_probas is not None:
            return (K.sum(K.log([old_proba])) + K.log([new_probas]))/number_of_elements
        else:
            return (K.sum(K.log([old_proba])))/len(old_proba)
         
    
    def _get_top_k(self, dense_out):
        dense_out_squeezed = self.squeeze_dimentions(dense_out)
        top_k_indexes = tf.argsort(dense_out_squeezed)[-self.beam_width:]
        top_k_probs = tf.sort(dense_out_squeezed)[-self.beam_width:]
        return top_k_indexes, top_k_probs
        
    def predict_first_k_word(self, input_sentence, tokenizer_inputs, max_len_input):
        x = tokenizer_inputs.texts_to_sequences([input_sentence])
        x = pad_sequences(x, maxlen=max_len_input)
        decoder_initial_state = self.network.get_decoder_initial_return_states(x, predict=True)
        docoder_input = np.full((1,1), self.network.sos_token)
        decoder_out = self.decoder.predict([decoder_initial_state[0], decoder_initial_state[1], docoder_input])
        decoder_dense_out = decoder_out[0]
        new_internal_state = [decoder_out[1]]
        try:
            new_internal_state += [decoder_out[2]]
        except:
            pass
            
        top_k_indexes, top_k_probs = self._get_top_k(decoder_dense_out)
        return top_k_indexes, top_k_probs, new_internal_state
    
    def get_next_words(self, prev_words:list, prev_probs:list, decoder_initial_state_list):
        next_words_list = []
        next_proba_list = []
        next_internal_state = []
        _local_proba_list = []
        _local_dense_out = []
        old_words_reference_index = []
    
        
        if not isinstance(decoder_initial_state_list[0], list):
            decoder_initial_state_list = [decoder_initial_state_list]*len(prev_words)
        
        for i, word in enumerate(prev_words):
            decoder_initial_state = decoder_initial_state_list[i]
            
            if tf.is_tensor(word):
                word = word.numpy()
                
            prev_prob = prev_probs[i]
            
            docoder_input = np.full((1,1), word)
            decoder_out = self.decoder.predict([decoder_initial_state[0],
                                                decoder_initial_state[1],
                                                docoder_input])
            decoder_dense_out = decoder_out[0]
            new_internal_state = [decoder_out[1]]
            try:
                new_internal_state += [decoder_out[2]]
            except:
                pass
            
            next_internal_state.append(new_internal_state)
            decoder_dense_out_squeezed = self.squeeze_dimentions(decoder_dense_out)
            _local_dense_out.append(decoder_dense_out_squeezed)
            new_combined_proba = self.calculate_combined_probability(np.array([prev_prob]), decoder_dense_out_squeezed)
            _local_proba_list.append(new_combined_proba)
            
        #next_proba_list = 50,50,50
        vocab_size = max(_local_proba_list[0].shape)
        #Concat all k probas to get top 3
        next_proba_concat = tf.concat(_local_proba_list, axis=0)
        top_k_indexes, top_k_probs = self._get_top_k(next_proba_concat)
        
        for i, ind in enumerate(top_k_indexes):
            which_old_word_in_input_list = ind//vocab_size
            old_words_reference_index.append(which_old_word_in_input_list.numpy())
            which_index = ind%vocab_size
            next_words_list.append([prev_words[which_old_word_in_input_list], which_index])
            next_proba_list.append([prev_probs[which_old_word_in_input_list], _local_dense_out[which_old_word_in_input_list][which_index]])
            #next_internal_state.append(_local_internal_states[which_old_word_in_input_list])
        return next_words_list, next_proba_list, next_internal_state, old_words_reference_index
    
    def _convert_array_to_list(self, arr):
        if tf.is_tensor(arr) or isinstance(arr, np.ndarray):
            arr = self.squeeze_dimentions(arr)
        return [val.numpy() for val in arr]
    
    def convert_to_numpy(self, val):
        if tf.is_tensor(val):
            val = val.numpy()
        return val
    
    def iterate(self):
        for seq_no in range(1, self.max_output_seq_len):
            #if seq_no == 2:
            #    import pdb; pdb.set_trace()
            local_word_list = [word_list[-1] for word_list in self.final_words.values()]
            local_proba_list = [proba_list for proba_list in self.final_probs.values()]
            local_internal_states = [state for state in self.final_internal_states.values()]

            local_word_list, local_proba_list, local_internal_states, old_words_reference_index = self.get_next_words(local_word_list, 
                                                                                           local_proba_list,
                                                                                          local_internal_states)

            local_word_list = [self.convert_to_numpy(lst[-1]) for lst in local_word_list]
            local_proba_list = [self.convert_to_numpy(lst[-1]) for lst in local_proba_list]


            for i, val in enumerate(old_words_reference_index):
                self.final_words[i] = self.final_words[val].copy()
                self.final_probs[i] = self.final_probs[val].copy()


            for i in range(self.beam_width):
                self.final_words[i].append(local_word_list[i])
                self.final_probs[i].append(local_proba_list[i])
                self.final_internal_states[i] = local_internal_states[i]
                if local_word_list[i] == self.network.eos_token:
                    self.final_sequence.append((self.final_words[i][:-1], 
                                                self.calculate_combined_probability(old_proba=self.final_probs[i]).numpy()))
                    
                    
    def return_best_seq(self, k=1):
        final_probas = np.zeros((1, len(self.final_sequence)))
        for i, seq in enumerate(self.final_sequence):
            final_probas[0][i] = seq[1]
        
        k_best_index = np.argsort(np.ravel(final_probas))[-k:]
        k_best_seq = [self.final_sequence[ind][0] for ind in k_best_index]
        return_seq = [seq[:seq.index(self.network.eos_token) if self.network.eos_token in seq else len(seq)] for seq in k_best_seq]
        return return_seq
        
        
    def decode(self, input_sentence:str, tokenizer_inputs, max_len_input, best_k=1):
        self.final_words = {k:[] for k in range(self.beam_width)}
        self.final_probs = {k:[] for k in range(self.beam_width)}
        self.final_internal_states = {k:None for k in range(self.beam_width)}
        self.final_sequence = []
        next_words_list, next_proba_list, next_internal_state = self.predict_first_k_word(input_sentence=input_sentence, 
                                                                                   tokenizer_inputs=tokenizer_inputs,
                                                                                   max_len_input=max_len_input)
        
        
        if not isinstance(next_internal_state[0], list):
            next_internal_state = [next_internal_state]*3
        
        next_words_list = self._convert_array_to_list(next_words_list)
        next_proba_list = self._convert_array_to_list(next_proba_list)

        for i in range(self.beam_width):
            self.final_words[i].append(next_words_list[i])
            self.final_probs[i].append(next_proba_list[i])
            self.final_internal_states[i] = next_internal_state[i]
            
        self.iterate()
            
        return self.return_best_seq(k=best_k)
        
        
#Usage
bs = BeamSearch(network) #network is from encoder-decoder class instance
seq = bs.decode(input_sentence="are you going to market", tokenizer_inputs=tokenizer_inputs, max_len_input=max_len_input)
word2idx_outputs_inv = {v:k for k,v in word2idx_outputs.items()}
print(" ".join([word2idx_outputs_inv[word] for word in seq[0]]))

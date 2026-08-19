# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers,models,Input,Model
from tensorflow.keras.layers import Bidirectional,LSTM,Dense,Activation
from tensorflow.keras.layers import   Layer, Input,RNN, Conv1D, Flatten, Dense, Reshape,MaxPooling1D,DepthwiseConv1D
from tensorflow.keras import layers,Model
from __pycache__.utils import* 
from tensorflow.keras.layers import Add,concatenate,Rescaling,LayerNormalization,Dropout
from tensorflow import keras
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import re
import random
stop = stopwords.words('english')
from tensorflow.keras.models import load_model,Model 
import spacy
from sklearn.preprocessing import StandardScaler
import warnings;warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=RuntimeWarning) 
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', message='TensorFlow retracing') 

def get_model(students):

    input_stu = Input(students)
    
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_stu)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)    
    seq_stu = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)
    seq_stu = layers.Bidirectional(layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2
                                                , return_sequences=True))(seq_stu) 
    
    
    # input_sta = Input(statements)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(seq_stu)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    seq_sta = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)
    seq_sta = layers.Bidirectional(layers.LSTM(128,dropout=0.2,recurrent_dropout=0.2
                                                , return_sequences=True))(seq_sta) 
    
    attention_seq = layers.Attention()([seq_stu, seq_sta])
    query_encoding = layers.GlobalAveragePooling1D()(seq_stu)
    query_value_attention = layers.GlobalAveragePooling1D()(attention_seq)
    con = layers.Concatenate()([query_encoding, query_value_attention])
    decoder = layers.Dense(256)(con)
    decoder = layers.Activation('relu')(decoder)
    output = layers.Dense(24,activation = 'linear')(decoder)  
    model = Model(inputs = input_stu , outputs = output)   
    model.compile(loss = 'mse', optimizer = 'adam',metrics=['Accuracy'])
    return model
    


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, 1), initializer='random_normal', trainable=True,name='W')
        self.b = self.add_weight(shape=(1,), initializer='zeros', trainable=True,name='b')
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def without_stacked(students):
    
    input_stu = Input(students)
    
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_stu)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)    
    x = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)  
    x=AttentionLayer()(x)
    x=Flatten()(x)
    decoder = layers.Dense(256)(x)
    decoder = layers.Activation('relu')(decoder)
    output = layers.Dense(24,activation = 'linear')(decoder)
    
    model = Model(inputs = input_stu , outputs = output)
    model.compile(loss = 'mse', optimizer = 'adam',metrics=['Accuracy'])
    return model
    
def without_att(students):
    
    input_stu = Input(students)
    
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_stu)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = layers.Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)    
    x = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)  
    x = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)  
    x = layers.Bidirectional(layers.LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(x)  

    x=Flatten()(x)
    decoder = layers.Dense(256)(x)
    decoder = layers.Activation('relu')(decoder)
    output = layers.Dense(24,activation = 'linear')(decoder)
    
    model = Model(inputs = input_stu , outputs = output)
    model.compile(loss = 'mse', optimizer = 'adam',metrics=['Accuracy'])
    return model
    

def resentences(output):
    processed = re.sub("'", "", str(output))
    processed = re.sub("-", "", processed)
    processed = re.sub("[^a-zA-Z]", " ", processed)
    processed = " ".join([i.lower() for i in processed.split() if i.lower() not in stop])
    processed = processed.strip()
    return processed



def calculate_similarity_batch(sentences, prediction_input, batch_size=100):

    nlp = spacy.load("en_core_web_md")
    # Convert numpy.str_ to regular string
    prediction_input = str(prediction_input)

    # Split sentences into batches
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    
    # Process batches
    similarity_scores = []
    for batch in batches:
        docs = list(nlp.pipe(map(str, batch)))
        prediction_doc = nlp(prediction_input)
        
        batch_similarity_scores = [doc.similarity(prediction_doc) for doc in docs]
        similarity_scores.extend(batch_similarity_scores)
    return np.array(similarity_scores)


def find_similarity_senetence(prediction_input):
    
    Student = np.load('savedata/student.npy', allow_pickle=True)
    scores = calculate_similarity_batch(Student, prediction_input)
    max_index = np.argmax(scores)
    prediction_output=Student[max_index]

    prediction_output=prediction_output.strip()
    return prediction_output,max_index
    

def normalized_function(prediction_input):
    mode1=Mode1(prediction_input)
    output,indx= mode1.Predict(prediction_input)
    output=resentences(output)
    if not output:
        prediction_output,max_index=find_similarity_senetence(prediction_input)
        mode1=Mode1(prediction_output)
        output = mode1.Predict(prediction_input)
        output=resentences(output)
        return output ,max_index  
    else:
        return output,indx[0]
    
        
    
def reverse_strings(Statement_sequences):
    
    # Find the maximum length of the padded sequences
    max_length = max(len(seq) for seq in Statement_sequences)
    
    # Pad sequences
    Statement_padded = pad_sequences(Statement_sequences, maxlen=max_length, padding='post')
    
    # Convert padded sequences back to text using two lines
    original_text_sequences = [' '.join(map(str, seq)) for seq in Statement_padded if any(seq)]
    return original_text_sequences
    
    
def ouput_model(processed_Input,text_model,indx):  
    text_model=Text_model(indx)
    output=text_model.Predict(processed_Input)
    return output
     


def Extraxt_sequence(prediction_input,fea_extr,text_model):

    processed_input = re.sub("'", "", str(prediction_input))
    processed_input = re.sub("-", "", processed_input)
    processed_input = re.sub("[^a-zA-Z]", " ", processed_input)
    processed_input = " ".join([i.lower() for i in processed_input.split() if i.lower() not in stop])
    processed_input = processed_input.strip()

    Processed_input,indx=normalized_function(prediction_input)
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts([processed_input])  # Note: Pass a list containing the single processed input
    students_sequences = tokenizer.texts_to_sequences([processed_input])
    
    # Pad sequence to a fixed length
    max_length = max(len(seq) for seq in students_sequences)
    students_padded = pad_sequences(students_sequences, maxlen=16, padding='post')
  
    scaler_statement = MinMaxScaler()
    students_padded_scaled = scaler_statement.fit_transform(students_padded)

    expands=np.expand_dims(students_padded_scaled,axis=-1)
    
    processed_input = fea_extr.predict(expands,verbose=0)
    processed_input_2d = processed_input.reshape(-1, 1)

    # Use MinMaxScaler for normalization
    processed__input= reverse_strings(processed_input_2d)
    
    emotion_model=ouput_model(processed__input,text_model,indx)

    return Processed_input,emotion_model

def Test(prediction_input,fea_extr,text_model):
    pred,emotion=Extraxt_sequence(prediction_input,fea_extr,text_model)
    return pred,emotion
    

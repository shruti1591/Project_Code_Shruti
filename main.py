# -*- coding: utf-8 -*-

from nltk.corpus import stopwords
import pandas as pd
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
stop = stopwords.words('english')
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import pad_sequences,to_categorical 
from tensorflow.keras.models import load_model,Model 
import tensorflow as tf 
import os
from pydub import AudioSegment
import numpy as np
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import LabelBinarizer
from text_features import*
from model import*
import pandas as pd
import re
from sklearn.preprocessing import LabelBinarizer
from audio_preprocessing import*
# from model import*
from SFO import*
    

# AN EFFICIENT AUTOMATED STUDENT’S PERSONALITY PREDICTION FROM THE TEXT AND AUDIO DATA

# def main():
    
    # read the data in dataset
data = pd.read_excel('dataset/data.xlsx')
df = data.drop(['label', 'Audio'], axis=1)
datas=df.copy()

# Combine all the preprocessing steps for remove punctuation,space ,number and quotation
processed_student_column = df['Student'].apply(lambda x: re.sub("'", "", str(x)))
processed_student_column = processed_student_column.apply(lambda x: re.sub("-", "", str(x)))
processed_student_column = processed_student_column.apply(lambda x: re.sub("[^a-zA-Z]", " ", str(x)))
processed_student_column = processed_student_column.apply(lambda x: " ".join([i.lower() for i in str(x).split() if i.lower() not in stop]))
processed_student_column = processed_student_column.str.strip()
df['Student'] = processed_student_column

# statement
processed_statement_column = df['Statement'].apply(lambda x: re.sub("'", "", str(x)))
processed_statement_column = processed_statement_column.apply(lambda x: re.sub("-", "", str(x)))
processed_statement_column = processed_statement_column.apply(lambda x: re.sub("[^a-zA-Z]", " ", str(x)))
processed_statement_column = processed_statement_column.apply(lambda x: " ".join([i.lower() for i in str(x).split() if i.lower() not in stop]))
processed_statement_column = processed_statement_column.str.strip()
df['Statement'] = processed_statement_column


# pad_sequences to add zeros to the sequences to make them all be the same length
# pad_sequence(uing the same length of our data)
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df['Student'])
student_sequences = tokenizer.texts_to_sequences(df['Student'])
df['Student'] = student_sequences
max_length = max(len(seq) for seq in student_sequences)
student_padded = pad_sequences(student_sequences, maxlen=max_length, padding='post')

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(df['Statement'])
Statement_sequences = tokenizer.texts_to_sequences(df['Statement'])
df['Statement'] = Statement_sequences
max_length = max(len(seq) for seq in Statement_sequences)
Statement_padded = pad_sequences(Statement_sequences, maxlen=max_length, padding='post')

# MinMaxScaler() function to normalize each feature by scaling the data to a range.
# Apply MinMax normalization student data
scaler_student = MinMaxScaler()
student_padded_scaled = scaler_student.fit_transform(student_padded)
# Apply MinMax scaling to 'statement data'
scaler_statement = MinMaxScaler()
statement_padded_scaled = scaler_statement.fit_transform(Statement_padded)


''' #####################   text_extraction-model   ####################'''
# ------------------------------------------------
#  Attention based Convolutional Stacked Bi-directional Long Term Short Memory

student=np.expand_dims(student_padded_scaled,axis=-1)
students=student.shape[1:]

model=get_model(students)
model.summary()
model.fit(student_padded_scaled,statement_padded_scaled,epochs=1)



''' ###########  text _output_model ##########'''


new_model = Model(inputs=model.input, outputs=model.layers[-2].output)
output_features = new_model.predict(student_padded_scaled)
output_shape=np.expand_dims(output_features,axis=-1)
output_shape=output_shape.shape[1:]


                                                                                                        
labels=data['label']
X_tr, X_te, y_tr, y_te = train_test_split(output_features, labels, test_size=0.2, random_state=42)

unique_tr, counts_tr = np.unique(y_tr, return_counts=True)

print("\nTraining labels:")
for label, count in zip(unique_tr, counts_tr):
    print(label, ":", count)

unique_te, counts_te = np.unique(y_te, return_counts=True)

print("\nTesting labels:")
for label, count in zip(unique_te, counts_te):
    print(label, ":", count)

# labels to using LabelBinarizer 0 and 1
lb=LabelBinarizer()
y_train=lb.fit_transform(y_tr)
y_test=lb.fit_transform(y_te)




from sklearn.preprocessing import LabelEncoder
import numpy as np

label_encoder = LabelEncoder()

y_tr_encoded = label_encoder.fit_transform(y_tr)
y_te_encoded = label_encoder.transform(y_te)




    # -----------------------------------------------------------------------------
    #%%
'''##################   audio     ##################'''

# Import other necessary libraries
data = pd.read_excel('dataset/data.xlsx')
scaler = MinMaxScaler()
files = "dataset/Audio"
gfxx_fea=[]

for i in os.listdir(files):
    file_path = os.path.join(files, i)
    
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)

    # Apply normalization
    normalized_audio = normalize_audio(audio)
    # normalized_audio.export(f"normalized_audio/{i}", format="wav")

    # Apply noise removal using Short Term Energy
    cleaned_audio = remove_noise(normalized_audio)
    # cleaned_audio.export(f"noise_removal/{i}", format="wav")

    
    # Gammatone frequency cepstral coefficients 
    # Extract GFCC features
    gfcc_features = extract_gfcc(cleaned_audio)
    normalized_gfcc=gfcc_features.flatten() 
    normalized_gfccs=normalized_gfcc[:3588]
    gfxx_fea.append(normalized_gfccs)
   
 
gfcc_features=np.asarray(gfxx_fea)
expands=np.expand_dims(gfcc_features,axis=-1)
inputs=expands.shape[1:]


''' #####################  audio_feature_mode ##################'''
# ------------deep convolutional recurrent neural network (DCRNN)----------------------
model=audios(inputs)
model.compile(loss = 'mse', optimizer = 'adam',metrics=['Accuracy'])
model.fit(gfcc_features,statement_padded_scaled,batch_size=32,epochs=10,verbose=0)


#reshape the audio_fea shape and ouput layers can be remove
new_audio_model = Model(inputs=model.input, outputs=model.layers[-3].output)
output_audio_features = new_audio_model.predict(gfcc_features)
output_audio_feature=output_audio_features.shape[1:]



''' #####################  audio_output_model##################'''

labels=data['label']
X_tr, X_te, y_tr, y_te = train_test_split(output_audio_features, labels, test_size=0.2, random_state=42)


#to using LabelBinarizer 0 to 1
lb=LabelBinarizer()
y_train=lb.fit_transform(audio_y_tr)
y_test=lb.fit_transform(audio_y_te)


"=============Lightweight Kimi-Delta Contrastive Neural Network (LiteKDC-Net )========"


from sklearn.preprocessing import LabelEncoder
import numpy as np

label_encoder = LabelEncoder()

audio_y_tr = label_encoder.fit_transform(audio_y_tr)
y_te_encoded = label_encoder.transform(audio_y_te)
# ape)

input_shape = audio_X_tr.shape[1:]
num_classes = len(np.unique(y_tr))



from liteKDC_Net import LiteKDCNetDual,train_dual_litekdc


model = LiteKDCNetDual(
    text_input_shape=X_tr.shape[1:],
    audio_input_shape=audio_X_tr.shape[1:],
    text_num_classes=len(np.unique(y_tr_encoded)),
    audio_num_classes=len(np.unique(audio_y_tr)),
    hidden_dim=128,
    proj_dim=64,
    num_heads=4
)

model = train_dual_litekdc(
    model=model,
    X_text_train=X_tr,
    y_text_train=y_tr_encoded,
    X_audio_train=audio_X_tr,
    y_audio_train=audio_y_tr,
    epochs=1,
    batch_size=32,
    lr=1e-3,
    contrastive_weight=0.1,
    temperature=0.07
)



model_1 = LiteKDCNetDual(text_input_shape=X_tr.shape[1:],audio_input_shape=None,
text_num_classes=len(np.unique(y_tr_encoded)),audio_num_classes=None)

model_1 = train_dual_litekdc(model=model,X_text_train=X_tr,y_text_train=y_tr_encoded,
    X_audio_train=None,y_audio_train=None)



model_2 = LiteKDCNetDual(text_input_shape=None,audio_input_shape=audio_X_tr.shape[1:],
    text_num_classes=None,audio_num_classes=len(p.unique(audio_y_tr)))

model_2 = train_dual_litekdc(model=model,X_text_train=None,y_text_train=None,
    X_audio_train=audio_X_tr,y_audio_train=audio_y_tr)








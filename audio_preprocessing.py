# -*- coding: utf-8 -*-


from tensorflow.keras.layers import Conv1D,Input,MaxPooling1D,SimpleRNN,Dense,Flatten
from tensorflow.keras import Model
from audio_preprocessing import*
from pydub import AudioSegment
import os
import numpy as np
from spafe.features.gfcc import gfcc
from sklearn.preprocessing import LabelBinarizer
from __pycache__.utils import* 
from sklearn.preprocessing import MinMaxScaler
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings;warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=DeprecationWarning) 
warnings.filterwarnings('ignore', category=RuntimeWarning) 
warnings.filterwarnings('ignore', category=FutureWarning) 
warnings.filterwarnings('ignore', message='TensorFlow retracing') 


def audios(inputs):
    print('inputs',inputs)
    input_shape=Input(inputs)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(input_shape)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = SimpleRNN(units=64, activation='tanh', return_sequences=True)(x)  
    x = MaxPooling1D(pool_size=(2, ), strides=(1,), padding='valid')(x) 
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x = Conv1D(32, kernel_size=(7,), strides=(2,), padding='same', activation="relu")(x)
    x=Flatten()(x)
    x=Dense(1,activation='linear')(x)
    model = Model(input_shape, x)
    return model


def resentences(output):
    processed = re.sub("'", "", str(output))
    processed = re.sub("-", "", processed)
    processed = re.sub("[^a-zA-Z]", " ", processed)
    processed = " ".join([i.lower() for i in processed.split() if i.lower() not in stop])
    processed = processed.strip()
    return processed


# this line normalizes the input audio
def normalize_audio(audio):
    normalized_audio = audio.normalize()
    return normalized_audio

  # threshold parameter with a default value of 10.
def remove_noise(audio, threshold=10):
    
    # This line extracts the audio samples from the input audio object (audio) and converts them into a NumPy array.
    samples = np.array(audio.get_array_of_samples())
    # This line calculates the energy of the audio by summing the squares of the audio samples.
    energy = np.sum(np.square(samples))
    
    # This line checks if the energy of the audio is below the specified threshold. If it is, the audio is considered as noise, and a new silent audio segment of the same duration 
    if energy < threshold:
        # If energy is below the threshold, consider it as noise and remove it
        return AudioSegment.silent(duration=len(audio))
    else:
        return audio

# Gammatone frequency cepstral coefficients 
#  The specific function gfcc is assumed to be a function that computes GFCC features, and it takes the audio samples and the frame rate as parameters.
def extract_gfcc(audio):
    samples = np.array(audio.get_array_of_samples())
    gfcc_feat = gfcc(samples, audio.frame_rate)
    return gfcc_feat


def audio_normalization(filename):
    Model=normal_path(filename)
    state,inx=Model.predition(filename)
    resent=resentences(state)
    return resent,inx

def reverse_strings(Statement_sequences):
    
    # Find the maximum length of the padded sequences
    max_length = max(len(seq) for seq in Statement_sequences)
    
    # Pad sequences
    Statement_padded = pad_sequences(Statement_sequences, maxlen=max_length, padding='post')
    
    # Convert padded sequences back to text using two lines
    original_text_sequences = [' '.join(map(str, seq)) for seq in Statement_padded if any(seq)]
    return original_text_sequences
    
def ouput_model(processed_Input,audio_model,inx):
    audio_models=Audio_model(inx)
    output=audio_models.Predict(processed_Input)
    return output
    
    
def extract_audio(filename,audio_fea,audio_model):
    audio = AudioSegment.from_wav(f'dataset/Audio/{filename}')
    nomalized_audio=normalize_audio(audio)
    Processed_input,inx=audio_normalization(filename)
    removenoise=remove_noise(nomalized_audio)
    gfcc_feat=extract_gfcc(removenoise)

    #flatten the features
    normalized_gfcc=gfcc_feat.flatten() 
    
    #same shape of gfccfeatures
    normalized_gfccs=normalized_gfcc[:3588]
    ex=np.expand_dims(normalized_gfccs,axis=0)
  
    processed_input = audio_fea.predict(ex,verbose=0)
    processed_input_2d = processed_input.reshape(-1, 1)

    # Use MinMaxScaler for normalization
    processed__input= reverse_strings(processed_input_2d)

    emotion_model=ouput_model(processed__input,audio_model,inx)
    return Processed_input,emotion_model


def test(filename,audio_fea,audio_model):
    pred,emotion=extract_audio(filename,audio_fea,audio_model)
    return pred,emotion

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import read

# Assuming the normalize_audio and remove_noise functions are already defined
def plot_individual_audio_waveforms(file_path):
    # Load the original audio file
    audio = AudioSegment.from_wav(file_path)
    
    # Convert the audio to numpy array for plotting
    original_audio = np.array(audio.get_array_of_samples())
    
    # Apply normalization
    normalized_audio = normalize_audio(audio)
    normalized_audio_ = np.array(normalized_audio.get_array_of_samples())
    
    # Apply noise removal
    cleaned_audio = remove_noise(normalized_audio)  # Assuming remove_noise outputs an AudioSegment object
    cleaned_audio = np.array(cleaned_audio.get_array_of_samples())  # Convert cleaned audio to numpy array
    
    # Create a time axis based on audio length
    time = np.linspace(0, len(original_audio) / audio.frame_rate, num=len(original_audio))
    
    # Plot the original audio
    plt.figure(figsize=(10, 4))
    plt.plot(time, original_audio, label='Original Audio', color='blue')
    # plt.title('Original Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()
    plt.savefig("original_audio_waveform.png")  # Save as PNG
    plt.close()  # Close the plot to free up memory

    # # Plot the normalized audio
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, normalized_audio_, label='Normalized Audio', color='green')
    # plt.title('Normalized Audio')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.show()

    # Plot the cleaned audio
    plt.figure(figsize=(10, 4))
    plt.plot(time, cleaned_audio, label='Cleaned Audio', color='red')
    # plt.title('Cleaned Audio')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()
    plt.savefig("original_Cleaned_waveform.png")  # Save as PNG
    plt.close()  # Close the plot to free up memory


# Example usage with a sample audio file
file_path = "Audio/audio0002.wav"
plot_individual_audio_waveforms(file_path)

# Define the remove_noise function with proper audio handling
def remove_noise(audio, threshold=10):
    # This line extracts the audio samples from the input audio object (audio) and converts them into a NumPy array.
    samples = np.array(audio.get_array_of_samples())
    # This line calculates the energy of the audio by summing the squares of the audio samples.
    energy = np.sum(np.square(samples))
    
    # This line checks if the energy of the audio is below the specified threshold. If it is, the audio is considered as noise, and a new silent audio segment of the same duration 
    if energy < threshold:
        # If energy is below the threshold, consider it as noise and remove it
        return AudioSegment.silent(duration=len(audio))
    else:
        return audio

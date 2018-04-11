import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from musicMethods import textSplit, bodyToInt, intToNote
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Settings
filename = 'Data/abc.txt'
weights_filename = "Checkpoints/music_0.2889.hdf5"
seq_length = 200 #Length of training sequences to feed into the network
creativity = .6
minN = 0
maxN = 15

#Defs
start_seq = []
raw_text = []
text = []
for i in textSplit(filename):
    text.append(i[1])
for i in text:
    start = []
    for j in bodyToInt(i):
        raw_text.append(j)
        start.append(j)
    start_seq.append(start)

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
n_chars = len(raw_text)
n_vocab = len(chars)
dataX = []
dataY = []

#Breaking the text into patterns to feed into network
for i in range(0, n_chars - seq_length):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

#print("Total Characters: ", n_chars)
#print("Total Vocab: ", n_vocab)
#print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length, 1)) #Reshaping the data for Keras
X = X / float(n_vocab) #Normalizing the data
y = np_utils.to_categorical(dataY) #Something about hot encoding?

#Defining the Model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation = "softmax")) #Output layer
model.compile(loss = "categorical_crossentropy", optimizer = "adam")

def train(e, load = True):
    """Trains the network"""
    #Creating checkpoint system
    if(load):
        model.load_weights(weights_filename)
    filepath="Checkpoints/music_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]

    # Do the thing!
    model.fit(X, y, epochs=e, batch_size=128, callbacks=callbacks_list)

def generate(leng, log = True):
    """Generates text"""
    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    #Pick a random seed
    start = np.random.randint(0, len(start_seq) - 1)
    pattern = start_seq[start]
    if(len(pattern) < 200):
        try:
            plen = len(pattern)
            for i in range(plen, 200):
                pattern.append(start_seq[start + 1][i - plen])
        except:
            plen = len(pattern)
            for i in range(plen, 200):
                pattern.append(start_seq[0][i - plen])
    if(len(pattern) > 200):
        pattern = pattern[len(pattern) - 200:]
    for i in range(0, len(pattern)):
        pattern[i] = char_to_int[pattern[i]]
    pattern_output = []
    for i in range(len(pattern)):
        pattern_output.append(int_to_char[pattern[i]])
    output_raw = []

    #Generate characters
    i = 0
    result = ""
    while(i <= leng or char_to_int[result] != char_to_int["\n"]):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        m = max(prediction[0])
        choices = []
        for j in prediction[0]:
            if(j / m >= creativity):
                choices.append(j)
        index = prediction[0].tolist().index(np.random.choice(choices))
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        output_raw.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
        i += 1
    prev = 0
    output = pattern_output + output_raw
    for i in range(len(output)): #Filters the output so that it's actually readable in .abc
        if(output[i] == "(~)"):
            output[i] = "~"
        elif(output[i] in ["(0)", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)"]):
            output[i] = output[i][1]
        elif(output[i] in ['-15', '-14', '-13', '-12', '-11', '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']): #Yeah, this is a stupid way to do this-- so what???
            note = int(output[i])
            if(note + prev <= minN):
                prev = minN
            elif(note + prev >= maxN):
                prev = maxN
            output[i] = intToNote(note + prev)
            prev = note
    output_final = ""
    for j in output:
        output_final += j
    if(log):
        print(output)
    else:
        return output

#train(50, True)
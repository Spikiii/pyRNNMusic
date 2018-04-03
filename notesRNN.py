import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam
from musicMethods import textSplit, bodyToInt, intToNote
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#Settings
filename = 'Data/abc.txt'
weights_filename = "Checkpoints/notes_2.4917.hdf5"
seq_length = 20 #Length of training sequences to feed into the network
creativity = 0.8
learning_rate = 0.001
drop_out = 0
minN = 0
maxN = 15

#Defs
intData = [] #Interval data
strData = "" #Structure data
text = []
for i in textSplit(filename):
    text.append(i[1])
for i in text:
    for j in bodyToInt(i)[1]:
        strData += j
    for j in bodyToInt(i)[0]:
        intData.append(j)
    strData += "\n"

noteOccs = [] #Lists indexes of Note Occurances
for i in range(len(strData)):
    if(strData[i] == "~"):
        noteOccs.append(i)

intChars = range(-15, 15)
strChars = sorted(list(set(strData)))
char_to_int = dict((c, i) for i, c in enumerate(strChars))
int_to_char = dict((i, c) for i, c in enumerate(strChars))
n_int_chars = len(intData)
n_str_chars = len(strData)
n_int_vocab = len(intChars)
n_str_vocab = len(strChars)

intDataX = []
strDataX = []
dataX = []
dataY = []

#Breaking the text into patterns to feed into network
for i in range(0, n_int_chars - seq_length):
    int_seq_in = intData[i:i + seq_length]
    str_seq_in = strData[noteOccs[i]:noteOccs[i] + seq_length]
    int_seq_out = intData[i + seq_length]
    intDataX.append(int_seq_in)
    strDataX.append([char_to_int[char] for char in str_seq_in])
    dataY.append(int_seq_out)

for i in range(len(intDataX)):
    for j in range(len(intDataX[i])):
        intDataX[i][j] = (intDataX[i][j] + len(intChars) / 2) / float(n_int_vocab)
for i in range(len(strDataX)):
    for j in range(len(strDataX[i])):
        strDataX[i][j] = (strDataX[i][j]) / float(n_int_vocab)
for i in range(len(intDataX)):
    temp = intDataX[i]
    for j in strDataX[i]:
        temp.append(j)
    dataX.append(temp)

n_patterns = len(dataX)

#print("Total Characters: ", n_chars)
#print("Total Vocab: ", n_vocab)
#print("Total Patterns: ", n_patterns)

X = np.reshape(dataX, (n_patterns, seq_length * 2, 1)) #Reshaping the data for Keras
y = []
for i in dataY:
    temp = [0] * len(intChars)
    temp[i] = 1
    y.append(temp)
y = np.array(y)

#Defining the Model
model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(256, return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(256))
model.add(Dropout(drop_out))
model.add(Dense(y.shape[1], activation = "softmax")) #Output layer
optim = Adam(lr = learning_rate)
model.compile(loss = "categorical_crossentropy", optimizer = optim)

def train(e, load = True):
    """Trains the network"""
    #Creating checkpoint system
    if(load):
        model.load_weights(weights_filename)
    filepath="Checkpoints/notes_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]

    # Do the thing!
    model.fit(X, y, epochs = e, batch_size = 256, callbacks = callbacks_list)

def generate(seed_raw, log = True):
    """Generates text"""
    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    #Find note occurances in the piece of structure
    noteOccs = []
    for i in range(len(seed_raw)):
        if (seed_raw[i] == "~"):
            noteOccs.append(i)

    #Filters seed_raw into normalized numbers
    strSeed_raw = []
    for i in seed_raw:
        strSeed_raw.append(char_to_int[i] / len(strChars))

    #Start from the seed
    strSeed = [j / len(strChars) for j in strSeed_raw[:seq_length]]
    intSeed = [(j + len(intChars) / 2) / len(intChars) for j in intDataX[int(np.random.random_sample() * len(intDataX))][:seq_length]]
    pattern = strSeed + intSeed
    output = []
    #Generate characters
    i = 0
    while(i <= len(noteOccs) - 1):
        x = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(x, verbose = 0)
        m = max(prediction[0])
        choices = []
        for j in prediction[0]:
            if(j / m >= creativity):
                choices.append(j)
        index = prediction[0].tolist().index(np.random.choice(choices))
        result = index / len(intChars)
        str_seq_in = pattern[seq_length:] + [strSeed_raw[noteOccs[i]]]
        int_seq_in = pattern[:seq_length] + [result]
        output.append(result)
        pattern = int_seq_in[1:] + str_seq_in[1:]
        i += 1
    #This part is good
    notes = []
    prev = 0
    for i in output:
        j = i * len(intChars) - len(intChars) / 2
        if(j + prev <= minN):
            prev = minN
        if(j + prev >= maxN):
            prev = maxN
        notes.append(intToNote(j + prev))
        prev = j + prev
    if(log):
        print(notes)
    else:
        return notes

train(50, True)
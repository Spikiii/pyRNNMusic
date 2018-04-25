"""This is a version of musicRNN built to run specifically off of a raspberry pi.
   It just contains the generation part of the code, and none of the training."""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from musicMethods import intToNote
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#Settings
weights_filename = "Checkpoints/music_2.3903.hdf5"
seeds_filename = "Data/music_seeds.txt"
seq_length = 200 #Length of training sequences to feed into the network
creativity = .4
minN = 0
maxN = 15
print("Imports Loaded")

#Defs
char_to_int = {'\n': 0, ' ': 1, '"': 2, '#': 3, '&': 4, '(': 5, '(0)': 6, '(1)': 7, '(2)': 8, '(3)': 9, '(4)': 10, '(5)': 11, '(6)': 12, '(7)': 13, '(8)': 14, '(9)': 15, '(~)': 16, ')': 17, '-': 18, '-1': 19, '-10': 20, '-11': 21, '-12': 22, '-13': 23, '-14': 24, '-15': 25, '-16': 26, '-17': 27, '-19': 28, '-2': 29, '-21': 30, '-22': 31, '-24': 32, '-26': 33, '-3': 34, '-4': 35, '-5': 36, '-6': 37, '-7': 38, '-8': 39, '-9': 40, '.': 41, '/': 42, '0': 43, '1': 44, '10': 45, '11': 46, '12': 47, '13': 48, '14': 49, '15': 50, '16': 51, '17': 52, '19': 53, '2': 54, '20': 55, '21': 56, '22': 57, '23': 58, '24': 59, '25': 60, '26': 61, '3': 62, '4': 63, '5': 64, '6': 65, '7': 66, '8': 67, '9': 68, ':': 69, '<': 70, '=': 71, '>': 72, 'H': 73, 'I': 74, 'J': 75, 'K': 76, 'L': 77, 'M': 78, 'O': 79, 'P': 80, 'Q': 81, 'R': 82, 'S': 83, 'V': 84, '[': 85, '\\': 86, ']': 87, '_': 88, 'h': 89, 'i': 90, 'k': 91, 'l': 92, 'm': 93, 'n': 94, 'o': 95, 'p': 96, 'r': 97, 's': 98, 't': 99, 'u': 100, 'v': 101, 'w': 102, 'x': 103, 'y': 104, 'z': 105, '{': 106, '|': 107, '}': 108}
int_to_char = {0: '\n', 1: ' ', 2: '"', 3: '#', 4: '&', 5: '(', 6: '(0)', 7: '(1)', 8: '(2)', 9: '(3)', 10: '(4)', 11: '(5)', 12: '(6)', 13: '(7)', 14: '(8)', 15: '(9)', 16: '(~)', 17: ')', 18: '-', 19: '-1', 20: '-10', 21: '-11', 22: '-12', 23: '-13', 24: '-14', 25: '-15', 26: '-16', 27: '-17', 28: '-19', 29: '-2', 30: '-21', 31: '-22', 32: '-24', 33: '-26', 34: '-3', 35: '-4', 36: '-5', 37: '-6', 38: '-7', 39: '-8', 40: '-9', 41: '.', 42: '/', 43: '0', 44: '1', 45: '10', 46: '11', 47: '12', 48: '13', 49: '14', 50: '15', 51: '16', 52: '17', 53: '19', 54: '2', 55: '20', 56: '21', 57: '22', 58: '23', 59: '24', 60: '25', 61: '26', 62: '3', 63: '4', 64: '5', 65: '6', 66: '7', 67: '8', 68: '9', 69: ':', 70: '<', 71: '=', 72: '>', 73: 'H', 74: 'I', 75: 'J', 76: 'K', 77: 'L', 78: 'M', 79: 'O', 80: 'P', 81: 'Q', 82: 'R', 83: 'S', 84: 'V', 85: '[', 86: '\\', 87: ']', 88: '_', 89: 'h', 90: 'i', 91: 'k', 92: 'l', 93: 'm', 94: 'n', 95: 'o', 96: 'p', 97: 'r', 98: 's', 99: 't', 100: 'u', 101: 'v', 102: 'w', 103: 'x', 104: 'y', 105: 'z', 106: '{', 107: '|', 108: '}'}
raw_file = open(seeds_filename)
seeds = []
for line in raw_file: #Splitting the seeds text up
    temp = line.split(",")
    for i in range(0, len(temp) - 1):
        temp[i] = int(temp[i])
    seeds.append(temp[:len(temp) - 1])

#Defining the Model
model = Sequential()
model.add(LSTM(256, input_shape = (200, 1), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(109, activation = "softmax")) #Output layer
model.compile(loss = "categorical_crossentropy", optimizer = "adam")
print("Model Initialized")

def generate(leng, log = True):
    """Generates text"""
    #Load the network weights
    model.load_weights(weights_filename)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

    #Pick a random seed
    pattern = seeds[np.random.randint(0, len(seeds) - 1)]
    pattern_output = []
    for i in range(len(pattern)):
        pattern_output.append(int_to_char[pattern[i]])
    output_raw = []

    #Generate characters
    i = 0
    result = ""
    while(i <= leng or char_to_int[result] != char_to_int["\n"]):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(char_to_int))
        prediction = model.predict(x, verbose=0)
        m = max(prediction[0])
        choices = []
        for j in prediction[0]:
            if(j / m >= 1 - creativity):
                choices.append(j)
        index = prediction[0].tolist().index(np.random.choice(choices))
        result = int_to_char[index]
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
        print(output_final)
    else:
        return output_final

generate(200, True)
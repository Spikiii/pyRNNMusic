"""
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
"""

#Settings
filename = "Data/abc.txt"

#Defs
notes = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]
X = []

def textSplit(filename):
    """Splits the text into header and notes, reads from the file, and returns 
       the split array"""
    #Defs
    raw_file = open(filename)
    raw_text = []
    X = []
    songs = []
    prev_end = 0

    for line in raw_file:
        raw_text.append(line)

    for i in range(len(raw_text)):
        if(raw_text[i] == "~\n"):
            songs.append(raw_text[prev_end:i])
            prev_end = i
    songs.append(raw_text[prev_end:i])

    for song in songs:
        for l in range(len(song)):
            if(song[l][0] == "K"):
                header = ""
                body = ""
                for i in range(l + 1):
                    header += song[i]
                for i in range(l + 1, len(song)):
                    body += song[i]
                X.append([header, body])
    return X

def intToNote(i):
    """Converts notes from an integer value to their abc notation counterpart, 
       in integer half-steps above / below middle C"""
    i = int(i)
    if(0 <= i <= 11):
        return notes[i]
    elif(11 < i <= 23):
        return notes[i - 12].lower()
    elif(23 < i):
        ap = -1
        while(i > 11):
            ap += 1
            i -= 12
        return notes[i].lower() + ("'" * ap)
    else: #Below middle C
        ap = 0
        while(i < 0):
            ap += 1
            i += 12
        return notes[i] + ("," * ap)

def noteToInt(n):
    """Converts from an abc notation note to it's integer number of half-steps
       above / below middle C"""
    i = 0
    if(n[len(n) - 1] == ","):
        while(n[len(n) - 1] == ","):
            n = n[:len(n) - 1]
            i -= 12
    elif(n[len(n) - 1] == "'"):
        while(n[len(n) - 1] == "'"):
            n = n[:len(n) - 1]
            i += 12
    try:
        return i + notes.index(n)
    except:
        try:
            return i + notes.index(n.upper()) + 12
        except:
            print("error")

def bodyToInt(text):
    """Converts notes to the intervals between the notes. Feed in the body of a
       piece of text that has been run through textSplit()"""
    bNotes = []
    bStructure = []
    for i in range(len(text)):
        note = ""
        if(text[i] == "^"): #If there's a sharp of flat
            note += text[i]
            note += text[i + 1]
            i += 2
            try:
                while text[i] == "'" or text[i] == ",":
                    note += text[i]
                    i += 1
                bNotes.append(note)
                bStructure.append("~")
            except:
                bNotes.append(note)
                bStructure.append("~")
        elif(text[i].upper() in notes): #If it's just a note
            note += text[i]
            i += 1
            try:
                while text[i] == "'" or text[i] == ",":
                    note += text[i]
                    i += 1
                bNotes.append(note)
                bStructure.append("~")
            except:
                bNotes.append(note)
                bStructure.append("~")
        elif(text[i] != "'" and text[i] != ","): #Anything else
            bStructure.append(text[i])
    for i in range(len(bNotes)): #Turns notes into numbers
        bNotes[i] = noteToInt(bNotes[i])
    iNotes = [bNotes[0]]
    for i in range(1, len(bNotes)): #Turns numbers into intervals between notes
        iNotes.append(bNotes[i] - bNotes[i - 1])
    return iNotes, bStructure
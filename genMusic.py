from titleRNN import generate as titleGen
from headerRNN import generate as headerGen
from structureRNN import generate as structureGen
from notesRNN import generate as notesGen

def genSong(leng):
    output = ""
    output += headerGen(200, False) + "\n"
    print(output)
    structure = structureGen(leng, False)
    notes = notesGen(structure, False)
    noteOccs = []  # Lists indexes of Note Occurances
    for i in range(len(structure)):
        if (structure[i] == "~"):
            noteOccs.append(i)
    notesPos = 0
    output = ""
    for j in structure:
        if(j == "~"):
            output += notes[notesPos]
            notesPos += 1
        else:
            output += j
    print(output)

genSong(200)
#while(input("::") != "x"):
#    genSong(200)
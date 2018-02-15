
#Defs
notes = ["C", "^C", "D", "^D", "E", "F", "^F", "G", "^G", "A", "^A", "B"]

def noteConvert(i):
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
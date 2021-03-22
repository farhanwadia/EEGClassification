from pythonosc import dispatcher, osc_server
import numpy as np
import tkinter as tk
import itertools
import asyncio
import threading
import random
import os
import sys

def swap(a, b):
    return b, a

def bothEven(a, b):
    return a % 2 == 0 and b % 2 == 0

def bothOdd(a, b):
    return a % 2 == 1 and b % 2 == 1

def stopProgram(window):
    window.destroy()
    sys.exit(0)

def incrementFileName(save_path, f_name):
# Returns a full csv path with "_(i)" at the end of the filename for the first i where the file didn't already exist
    i = 1
    full_path = save_path + "\\" + f_name + "_(" + str(i) + ").csv"
    while os.path.exists(full_path):
        i = i + 1
        full_path = save_path + "\\" + f_name + "_(" + str(i) + ").csv"
        if not (os.path.exists(full_path)):
            break
    return full_path

def generateFullPath(save_path, subject, trialNum, targetSide, leftFreq, rightFreq, restRecordingTime = None):
# Creates the full path of where to save each trial
# If the file already exists, a number is appended to the end to prevent
# original from being overwritten

    if restRecordingTime is None:
        targetFreq = chooseTarget(targetSide, leftFreq, rightFreq)
        f_name = subject + "_T" + str(trialNum) + "_Target_" + targetSide[0].upper() + str(targetFreq) + \
                 "_L" + str(leftFreq) + "R" + str(rightFreq)

        if not(os.path.exists(save_path + "\\" + f_name + ".csv")):
            return save_path + "\\" + f_name + ".csv"
        else:
            return incrementFileName(save_path, f_name)
    else:
        f_name = subject + "_T" + str(trialNum) + "_Rest_" + str(restRecordingTime) + "s"
        if not(os.path.exists(save_path + "\\" + f_name + ".csv")):
            return save_path + "\\" + f_name + ".csv"
        else:
            return incrementFileName(save_path, f_name)

def chooseTarget(targetSide, leftItem, rightItem):
# Returns leftItem or rightItem according to targetSide
    if targetSide == 'left':
        return leftItem
    elif targetSide == 'right':
        return rightItem
    else:
        return None

def createWindow(title, backgroundColor, fullScreen=False): 
# Returns a tk.tk() object with given title and background colour
    window = tk.Tk()
    window.title(title)
    if fullScreen:
        window.state('zoomed')
    window.configure(bg=backgroundColor)
    return window

class FlashingRectangle:
# Creates a flashing rectangle given parent tkinter object named master, width, height,
# outline color, fill color, flash color, (all colours are tkinter colour strings),
# and flashing frequency (Hz). Choose frequencies that 1000 cleanly divides by.

    def __init__(self, master, length, outlineStr, fillStr, flashStr, flashFreq):
        self.canv = tk.Canvas(master, width=length, height=length, highlightthickness=0)
        self.rect = self.canv.create_rectangle(0, 0, length, length, outline=outlineStr, fill=fillStr)
        self.canv.focus()
        self.canv.grid()
        self.fillColor = fillStr
        self.flashColor = flashStr
        self.flashFreq = flashFreq

        self.flash()  # comment out this line to stop the flashing

    def flash(self):
        self.canv.itemconfigure(self.rect, fill=self.flashColor)  # Flash
        self.fillColor, self.flashColor = swap(self.fillColor, self.flashColor)  # swap colours
        self.canv.after(int(1000 / self.flashFreq), self.flash)  # Call again after (1000 / flashFreq) ms

def createChessboardFrame(master, boardSize, dotSize, squaresPerSide, flashFreq, color1, color2, dotColor):
# Returns a tkinter Frame object of a flashing chessboard given parent Tkinter object named master,
# the board size, number of squares per side, colors, and flashing frequency (Hz)

    chessboard = tk.Frame(master=master, width=boardSize, height=boardSize, bd=0)
    for i in range(squaresPerSide):
        chessboard.rowconfigure(i, weight=0)
        for j in range(squaresPerSide):
            chessboard.columnconfigure(j, weight=0)  # Equal padding for all columns and rows

            # Create frame f with master as chessboard,
            # gridded at row i, column j of chessboard with 0 padding
            f = tk.Frame(master=chessboard)
            f.grid(row=i, column=j, ipadx=0, ipady=0)

            # Create flashing rectangle with local f as parent
            # Alternate starting colours for the squares in a chessboard pattern
            if bothEven(i + 1, j + 1) or bothOdd(i + 1, j + 1):
                FlashingRectangle(f, boardSize / squaresPerSide, '', color1, color2, flashFreq)
            else:
                FlashingRectangle(f, boardSize / squaresPerSide, '', color2, color1, flashFreq)

    # Create the dot in the centre of the board
    dotFrame = tk.Frame(master=chessboard, width=boardSize, height=boardSize)
    canv = tk.Canvas(master=dotFrame, width=dotSize, height=dotSize, highlightthickness=0)
    dot = canv.create_rectangle(0, 0, dotSize, dotSize, outline='', fill=dotColor)
    canv.focus()
    canv.grid()
    dotFrame.lift()
    dotTopLeft = (boardSize - dotSize) / 2
    dotFrame.place(x=dotTopLeft, y=dotTopLeft)

    # Return the board
    return chessboard

def _asyncio_thread(async_loop, ip, port, fullPath, t, trialNum, isRest):
    async_loop.run_until_complete(recordEEG(ip, port, fullPath, t,  async_loop, trialNum, isRest))

def do_tasks(async_loop, ip, port, fullPath, t, trialNum, isRest):
    """ Event-Handler starting the asyncio part. """
    threading.Thread(target=_asyncio_thread,
                     args=(async_loop, ip, port, fullPath, t, trialNum, isRest)).start()

def eeg_handler(unused_addr, args, ch1, ch2, ch3, ch4, ch5):
    recorded_signal.append([ch1, ch2, ch3, ch4])

def horseshoe(unused_addr, args, ch1, ch2, ch3, ch4):
    connection.append((ch1, ch2, ch3, ch4))

async def loop(delay):
    """Open server for delay time"""
    await asyncio.sleep(delay)

async def init_main(ip, port, dispatcher, t, async_loop):
    server = osc_server.AsyncIOOSCUDPServer((ip, port), dispatcher, async_loop)
    transport, protocol = await server.create_serve_endpoint()  # Create datagram endpoint and start serving
    await loop(t)  # Enter main loop of program
    transport.close()  # Clean up serve endpoint

async def recordEEG(ip, port, fullPath, t, async_loop, trialNum, isRest):
    from pythonosc import dispatcher
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")
    dispatcher.map("/muse/elements/horseshoe", horseshoe, "Horseshoe")
    recorded_signal.clear()
    connection.clear()
    await (init_main(ip, port, dispatcher, t, async_loop))
    connection_unique = list(set(connection))

    if (len(connection_unique) == 1) & (connection_unique[0] == (1,1,1,1)):
        good = True
        recorded_signal_file = np.array(recorded_signal, dtype='float64')
        np.savetxt(fullPath, recorded_signal_file, delimiter=",")
    else:
        print("Poor connection for " + fullPath + ". File was NOT created")
        if isRest:
            str = "Rest"
        else:
            str = "SSVEP"
        bad_trials.append([trialNum, str])
        good = False
    #return good

def initializeRecording(window, delay, EEGTime, fullPath, trialNum, isRest):
    ip = "127.0.0.1"
    port = 7000
    recorded_signal = []
    connection = []
    async_loop = asyncio.get_event_loop()
    window.after(delay * 1000,
                 lambda:do_tasks(async_loop, ip, port, fullPath, EEGTime, trialNum, isRest))

def getRestData(targetSide, EEGDelay, totalRestTime, trialNum, subject, savePath):
# Shows the rest window and side to stare at for upcoming trial while recording resting EEG
    backg = 'black'
    window = createWindow("Rest", backg, fullScreen=True)
    textColor = 'white'
    leftArrow = "⇦"
    rightArrow = "⇨"

    arrow = chooseTarget(targetSide, leftArrow, rightArrow)
    prompt = "Upcoming Trial:"
    restLabel = tk.Label(master=window, text="Rest", foreground=textColor, bg = backg, font='Calibri 128 bold')
    promptLabel = tk.Label(master=window, text=prompt, foreground=textColor, bg=backg, font='Calibri 48')
    arrowLabel = tk.Label(master=window, text=arrow, foreground=textColor, bg=backg, font='Calibri 128')

    restLabel.pack(expand=True, padx=10)
    promptLabel.pack(expand=True, padx=10)
    arrowLabel.pack(expand=True, padx=10)

    #Call code to record EEG for EEGTime seconds here
    EEGTime = totalRestTime - 2 * EEGDelay
    fullPath = generateFullPath(savePath, subject, trialNum, None, None, None, EEGTime)
    initializeRecording(window, EEGDelay, EEGTime, fullPath, trialNum, isRest = True)
    window.after(1000*EEGTime, lambda: window.destroy()) # destroy window after restTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window)) # kill all execution if user exits
    window.mainloop()

def getSSVEPData(targetFreq, otherFreq, targetSide, EEGDelay, EEGTime, trialNum, subject, savePath):
# Shows the required flashing chessboards and records the EEG signal

    # Parameters
    boardSize = 180
    dotSize = 10
    squaresPerSide = 6
    boardColor1 = 'gray15'
    boardColor2 = 'white'
    dotColor = 'red'
    textColor = 'white'
    backg = 'black'

    # Set up window
    window = createWindow("SSVEP Data Collection", backg, fullScreen=True)
    window.columnconfigure(2, weight=1)
    window.rowconfigure(2, weight=1)

    prompt = "Stare at the " + dotColor + " dot here"
    arrow = chooseTarget(targetSide, "⇦", "⇨")
    if targetSide == "left":
        leftText = prompt
        rightText = ""
        leftFreq = targetFreq
        rightFreq = otherFreq
    elif targetSide == "right":
        rightText = prompt
        leftText = ""
        rightFreq = targetFreq
        leftFreq = otherFreq

    # Child objects of window
    titleText = "Trial " + str(trialNum)
    titleLabel = tk.Label(master=window, text=titleText, foreground=textColor, bg=backg, font='Calibri 48 bold underline')
    arrowLabel = tk.Label(master=window, text=arrow, foreground=textColor, bg=backg, font='Calibri 140')
    
    leftLabelFrame = tk.LabelFrame(master=window, text=leftText, foreground=textColor, bg=backg)
    rightLabelFrame = tk.LabelFrame(master=window, text=rightText,foreground=textColor, bg=backg)
    
    # Chessboard frames are child objects of a parent label frame
    leftBoardFrame = createChessboardFrame(leftLabelFrame, boardSize, dotSize, squaresPerSide,
                                           leftFreq, boardColor1, boardColor2, dotColor)
    rightBoardFrame = createChessboardFrame(rightLabelFrame, boardSize, dotSize, squaresPerSide,
                                            rightFreq, boardColor1, boardColor2, dotColor)
    leftBoardFrame.pack(padx=10, pady=10)
    rightBoardFrame.pack(padx=10, pady=10)

    # Place label frames in window
    #trialNumLabel.grid(row=1, column=1, padx=10)
    titleLabel.grid(row=1, column=2, padx=20)
    leftLabelFrame.grid(row=2, column=1, padx=10)
    arrowLabel.grid(row=2, column=2, padx=200)
    rightLabelFrame.grid(row=2, column=3, padx=10)

    #Call code to record EEG for EEGTime seconds here
    fullPath = generateFullPath(savePath, subject, trialNum, targetSide, leftFreq, rightFreq)
    initializeRecording(window, EEGDelay, EEGTime, fullPath, trialNum, isRest = False)

    window.after(1000*(2*EEGDelay + EEGTime), lambda: window.destroy()) # destroy window after elapsed time
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window)) # kill all execution if user exits
    window.mainloop()
    
def runSingleTrial(targetFreq, otherFreq, targetSide, restTime, trialNum, subject, savePath):
# Shows the rest screen with instruction for the next trial.
# Collects EEG data during rest
# Then runs the trial and collects EEG data of the trial

    # Parameters
    EEGTime = 10  # length of each EEG recording
    EEGDelay = 1  # show the flashing checkerboards for EEGDelay seconds before starting and after ending recording

    getRestData(targetSide, EEGDelay, restTime, trialNum, subject, savePath)
    getSSVEPData(targetFreq, otherFreq, targetSide, EEGDelay, EEGTime, trialNum, subject, savePath)

def generateTrialTypes():
# Returns a 2D array of all 24 possible trials
# First column is the trialCode, second column is the left frequency,
# third column is the right frequency, and 4th column is the target side 
    frequencies = [4, 5, 8, 10]
    i = 1
    trialTypes = []
    for targetSide in ["left", "right"]:
        for perm in itertools.permutations(frequencies, 2):
            trialTypes.append([i, perm[0], perm[1], targetSide])
            i = i+1
    return trialTypes

def generateTrialOrder():
# Returns a 2D array of order to perform the trials in
# First column is the trial number, second column is the trial code
    random.seed(5)
    trialOrder = [i for i in range(1,25) for k in range(2)]
    random.shuffle(trialOrder)
    trials = [[i, trialOrder[i-1]] for i in range(1,49)]
    return trials

def generateRestTimes(trials, minRestTime, maxRestTime):
# Returns a 1D list of rest times between minRestTime and maxRestTime to use for each trial
    random.seed(5)
    return [random.randint(minRestTime, maxRestTime) for x in range(len(trials))]

def runMultipleTrials(startTrialNum, endTrialNum, subject, folderPath):
# Runs trials from startTrialNum to endTrialNum
    MIN = 1
    MAX = 48
    if startTrialNum not in range(MIN, MAX+1): return None
    if endTrialNum not in range(MIN, MAX+1): return None
    if startTrialNum > endTrialNum: return None
    
    trialTypes = generateTrialTypes()
    trials = generateTrialOrder()
    restTimes = generateRestTimes(trials, 10, 15)

    for i in range(startTrialNum-1, endTrialNum):
        trialNum = trials[i][0]
        trialCode = trials[i][1]
        leftFreq = trialTypes[trialCode - 1][1]
        rightFreq = trialTypes[trialCode - 1][2]
        targetSide = trialTypes[trialCode - 1][3]
        restTime = restTimes[i]

        if targetSide == 'left':
            targetFreq = leftFreq
            otherFreq = rightFreq
        elif targetSide == 'right':
            targetFreq = rightFreq
            otherFreq = leftFreq

        runSingleTrial(targetFreq, otherFreq, targetSide, restTime, trialNum, subject, folderPath)

def main():
# Enter your name and a folder path to save the EEG recording files to
# Run through trials 1 through 48 (you may want to go through it as 4 sets of 12 or 6*8 etc. rather than 1*48)
# If data collection was poor for any individual trials,
# edit startTrialNum and endTrialNum accordingly to redo those trials
    subject = "SD"
    folderPath = r"C:\Users\Stephanie DiNunzio\Documents\EEG Recordings\\"
    startTrialNum = 1
    endTrialNum = 48

    runMultipleTrials(startTrialNum, endTrialNum, subject, folderPath)
    print("Bad Trials:", bad_trials)

bad_trials = []
recorded_signal = []
connection = []
main()












from pythonosc import dispatcher, osc_server
import numpy as np
import tkinter as tk
import asyncio
import threading
import random
import os
import sys

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

def generateFullPath(save_path, subject, trialNum, description):
    f_name = subject + "_T" + str(trialNum) + " " + description
    if not(os.path.exists(save_path + "\\" + f_name + ".csv")):
        return save_path + "\\" + f_name + ".csv"
    else:
        return incrementFileName(save_path, f_name)

def createWindow(title, backgroundColor, fullScreen=False): 
# Returns a tk.tk() object with given title and background colour
    window = tk.Tk()
    window.title(title)
    if fullScreen:
        window.state('zoomed')
    window.configure(bg=backgroundColor)
    return window

def _asyncio_thread(async_loop, ip, port, fullPath, t):
    async_loop.run_until_complete(recordEEG(ip, port, fullPath, t,  async_loop))

def do_tasks(async_loop, ip, port, fullPath, t):
    """ Event-Handler starting the asyncio part. """
    threading.Thread(target=_asyncio_thread,
                     args=(async_loop, ip, port, fullPath, t)).start()

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

async def recordEEG(ip, port, fullPath, t, async_loop):
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
        good = False
    #return good

def initializeRecording(window, delay, EEGTime, fullPath):
    ip = "127.0.0.1"
    port = 7000
    #recorded_signal = []
    #connection = []
    async_loop = asyncio.get_event_loop()
    window.after(int(delay * 1000),
                 lambda:do_tasks(async_loop, ip, port, fullPath, EEGTime))

def showTrialWindow(windowTitle, leftEmoji, rightEmoji, emojiColor, trialNum, EEGTime, EEGDelay, description, savePath, subject):
    backg = 'black'
    textColor = 'white'
    window = createWindow(windowTitle, backg, fullScreen=True)
    window.columnconfigure(2, weight=1)
    window.rowconfigure(2, weight=1)

    #Make emoji labels
    if leftEmoji:
        leftHandLabel = tk.Label(master=window, text=leftEmoji, foreground=emojiColor, bg=backg, font='Calibri 150')
    else:
        leftHandLabel = tk.Label(master=window, text=rightEmoji, foreground=backg, bg=backg, font='Calibri 150')
    if rightEmoji:
        rightHandLabel = tk.Label(master=window, text=rightEmoji, foreground=emojiColor, bg=backg, font='Calibri 150')
    else:
        rightHandLabel = tk.Label(master=window, text=leftEmoji, foreground=backg, bg=backg, font='Calibri 150')

    trialLabel = tk.Label(master=window, text="Trial " + str(trialNum), foreground=textColor, bg=backg,
                          font='Calibri 48 bold underline')
    plusLabel = tk.Label(master=window, text="+", foreground=textColor, bg=backg, font='Calibri 150')

    #Place labels on window
    leftHandLabel.grid(row=2, column=1)
    rightHandLabel.grid(row=2, column=3)
    plusLabel.grid(row=2, column=2)
    trialLabel.grid(row=1, column=2)

    #Record EEG
    totalTime = EEGTime + EEGDelay + 1
    fullPath = generateFullPath(savePath, subject, trialNum, description)
    initializeRecording(window, EEGDelay, EEGTime, fullPath)

    window.after(int(1000 * totalTime), lambda: window.destroy())  # destroy window after totalTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window))  # kill all execution if user exits
    window.mainloop()

def showRestWindow(leftEmoji, rightEmoji, color, t):
    backg = 'black'
    textColor = 'white'
    window = createWindow("Rest", backg, fullScreen=True)
    restLabel = tk.Label(master=window, text="Rest", foreground=textColor, bg = backg, font='Calibri 128 bold')
    prompt = "Upcoming Trial \n" + leftEmoji + " + " + rightEmoji
    promptLabel = tk.Label(master=window, text=prompt, foreground=color, bg = backg, font='Calibri 72')
    restLabel.pack(expand=True)
    promptLabel.pack(expand=True)
    window.after(int(1000 * t), lambda: window.destroy())  # destroy window after restTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window))  # kill all execution if user exits
    window.mainloop()

def runAllTrials(subject, savePath):
    trials = [[1, "Open Both"],[2, "Close Both"],[3, "Open (valve)"], [4, "Close (valve)"]]
    emojis = [[1, "🖐", "🖐"],[2, "🤜", "🤛"], [3, "🚰", "🚰"], [4, "🚱", "🚱"]]
    colors = ['red', 'blue', 'DeepSkyBlue','orchid4']
    EEGTime = 8
    EEGDelay = 1.25
    numRepeats = 20

    for r in range(numRepeats):
        random.shuffle(trials)
        for i, trial in enumerate(trials):
            trialCode = trials[i][0]
            description = trials[i][1]
            color = colors[trialCode - 1]
            leftEmoji = emojis[trialCode - 1][1]
            rightEmoji = emojis[trialCode - 1][2]

            showRestWindow(leftEmoji, rightEmoji, color, random.randint(4, 8))
            showTrialWindow(description, leftEmoji, rightEmoji, color,4*r+i+1, EEGTime, EEGDelay, description, savePath, subject)
            #showRestWindow(random.randint(4,8))


# Change the subject and savePath
subject = "FW"
savePath = r"C:\Users\Farhan\Desktop\MI"
recorded_signal = []
connection = []
runAllTrials(subject, savePath)

#Notes:
# If you see open hands, imagine opening both your hands but do not actually move your hands
# If you see closed hands, imagine closing both your hands but do not actually move your hands
# If you see an open valve, constantly repeat open, open, open, open, open, open... in your head
# If you see a closed valve, constantly repeat close, close, close, close, close, close... in your head
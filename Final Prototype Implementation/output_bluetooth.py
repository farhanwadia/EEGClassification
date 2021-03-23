from pythonosc import dispatcher, osc_server
import numpy as np
import tkinter as tk
import asyncio
import threading
import random
import os
import sys
import pandas as pd
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import matlab.engine
import serial
import time
from firebase import firebase
from scipy.signal import butter, filtfilt


def butter_bandpass(channel, low_cutoff, high_cutoff, fs, order):
    nyq = fs/2
    # Get the filter coefficients
    b, a = butter(order, [low_cutoff/nyq, high_cutoff/nyq], btype='band', analog=False)
    y = filtfilt(b, a, channel)
    return y

def bandpassFilterChannels(data, low_cutoff, high_cutoff, fs, order):
    for col in range(data.shape[1]):
        #Filter each column with 5th order butterworth bandpass
        data[:, col] = butter_bandpass(data[:, col], 0.5, 40, 256, 5)
    return data


def process(data):
    fs = 256
    data = bandpassFilterChannels(data, 0.5, 40, fs, 5)
    data = data.transpose()
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    data = data.float()
    return data

class ANN_TS_3L(nn.Module):
    def __init__(self):
        super(ANN_TS_3L, self).__init__()
        self.name = "ANN_TS_3L"
        self.layer1 = nn.Linear((256*5)*4, 500)
        self.layer2 = nn.Linear(500, 200)
        self.layer3 = nn.Linear(200, 50)
        self.layer4 = nn.Linear(50, 2)
    def forward(self, img):
        flattened = img.reshape(-1, (256*5)*4)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        output = self.layer4(activation3)
        return output

def loadModels():
    torch.manual_seed(1)
    ANN = ANN_TS_3L()
    if use_cuda and torch.cuda.is_available():
        ANN = ANN.cuda()
        print("CUDA Available")

    model_path = path + "\\model_ANN_TS_3L_bs50_lr0-001_epoch72_iteration509_val-acc65-3333.pth"
    model = ANN

    if use_cuda and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, torch.device('cpu')))

    return ANN

def checkOutput(recorded_signals,model):
    recorded_signals = np.array(recorded_signals, dtype='float64')
    data = process(recorded_signals)
    if use_cuda and torch.cuda.is_available():
        data = data.cuda()
    out = model(data)
    out = out.max(1, keepdim=True)[1]
    return out


def createWindow(title, backgroundColor, fullScreen=False):
# Returns a tk.tk() object with given title and background colour
    window = tk.Tk()
    window.title(title)
    if fullScreen:
        window.state('zoomed')
    window.configure(bg=backgroundColor)
    return window

def stopProgram(window):
    window.destroy()
    sys.exit(0)

def _asyncio_thread(async_loop, ip, port, info):
    async_loop.run_until_complete(recordEEG(ip, port, info, async_loop))

def do_tasks(async_loop, ip, port, info):
    """ Event-Handler starting the asyncio part. """
    threading.Thread(target=_asyncio_thread,
                     args=(async_loop, ip, port, info)).start()

def initializeRecording(window, info):
    ip = "127.0.0.1"
    port = 7000
    async_loop = asyncio.get_event_loop()
    window.after(int(info[3] * 1000),
                 lambda:do_tasks(async_loop, ip, port, info))

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


async def recordEEG(ip, port, info, async_loop):
    from pythonosc import dispatcher
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler, "EEG")
    dispatcher.map("/muse/elements/horseshoe", horseshoe, "Horseshoe")
    recorded_signal.clear()
    connection.clear()
    desired, Emoji, description, EEGDelay, EEGTime = info
    await (init_main(ip, port, dispatcher, EEGTime, async_loop))
    connection_unique.clear()
    unique = list(set(connection))
    for i in unique:
        connection_unique.append(i)


def showPromptWindow():
    info = ["Prompt", False, "Think 'Open' or 'Close'", 1, 8]
    windowTitle, emoji, description, EEGDelay, EEGTime = info
    backg = 'black'
    textColor = 'white'
    emojiColor = 'DeepSkyBlue'
    window = createWindow(windowTitle, backg)

    # Make labels

    descriptionLabel = tk.Label(master=window, text=description, foreground=textColor, bg=backg, font='Calibri 72')

    # Place labels on window
    descriptionLabel.pack(expand=True)

    # Record EEG
    initializeRecording(window, info)

    totalTime = EEGTime + EEGDelay + 1
    window.after(int(1000 * totalTime), lambda: window.destroy())  # destroy window after totalTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window))  # kill all execution if user exits
    window.mainloop()

def showConnectionWindow():
    backg = 'black'
    textColor = 'white'
    window = createWindow("Bad Connection", backg)

    descriptionLabel = tk.Label(master=window, text="Poor Connection. Try Again.",
                                foreground=textColor, bg=backg, font='Calibri 72')
    descriptionLabel.pack(expand=True)

    totalTime = 3
    window.after(int(1000 * totalTime), lambda: window.destroy())  # destroy window after totalTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window))  # kill all execution if user exits
    window.mainloop()

def showResultWindow(result):
    t = 1.5
    backg = 'black'
    textColor = 'white'
    if result == 0:
        Emoji = "🚱"
        description = "Close"
    else:
        Emoji = "🚰"
        description = "Open"

    color = 'DeepSkyBlue'
    window = createWindow("Result", backg)
    resultLabel = tk.Label(master=window, text=description, foreground=textColor, bg = backg, font='Calibri 72 bold')
    emojiLabel = tk.Label(master=window, text=Emoji, foreground=color, bg = backg, font='Calibri 150')
    resultLabel.pack(expand=True)
    emojiLabel.pack(expand=True)
    window.after(int(1000 * t), lambda: window.destroy())  # destroy window after restTime seconds
    window.protocol("WM_DELETE_WINDOW", lambda: stopProgram(window))  # kill all execution if user exits
    window.mainloop()

def showOutput():
    model= loadModels()
    showPromptWindow()
    if (len(connection_unique) == 1) and (connection_unique[0] == (1, 1, 1, 1)):
        recorded_signal2 = recorded_signal[:1280]
        result = checkOutput(recorded_signal2, model)
        if result == 0:
            showResultWindow(result)
            return 0
        else:
            showResultWindow(result)
            return 1

    else:
        showConnectionWindow()
        return 2


use_cuda = True
recorded_signal = []
connection = []
connection_unique = []
path = os.getcwd()

fire = firebase.FirebaseApplication("https://eeg-valve-control-default-rtdb.firebaseio.com/", None)

i = True

while i:
    result = fire.get('/EEG_Valve_Control/start', '')
    if result == "\"1\"":
        print("Start")
        port = "COM6"  # This will be different for various devices and on windows it will probably be a COM port.
        bluetooth = serial.Serial(port, 9600)  # Start communications with the bluetooth unit
        print("Connected")
        bluetooth.flushInput()  # This gives the bluetooth a little kick
        out = 2
        while out == 2:  # send 5 groups of data to the bluetooth
            print("Ping")
            out = showOutput()
            print(out)
            bluetooth.write(str.encode(str(out)))  # These need to be bytes not unicode, plus a number
            input_data = bluetooth.readline()  # This reads the incoming data. In this particular example it will be the "Hello from Blue" line
            print(input_data.decode())  # These are bytes coming in so a decode is needed
            time.sleep(0.1)  # A pause between bursts
        bluetooth.close()  # Otherwise the connection will remain open until a timeout which ties up the /dev/thingamabob
        print("Done")
        fire.put('/EEG_Valve_Control', 'start','0')


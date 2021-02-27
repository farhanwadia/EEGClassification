import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
import scipy.fftpack
import pywt
import json

def loadCSV(filename):
    # Takes in a string of the csv file name where the EEG data is (# of measurements by 4 channels)
    # Returns the data as an np array
    data = np.loadtxt(filename, delimiter=',')
    return data

def getFiles(parentPath):
    # Returns all files in the folder and its subfolders as a list
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(parentPath):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    return listOfFiles

def makeMuseMontage():
    #Define electrode locations for the Muse
    #Get electrode positions from the 10-5 montage and extract only the positions Muse uses
    #Make a new montage for Muse 
    muse_chs = ['TP9', 'AF7', 'AF8', 'TP10']
    montage_1005 = mne.channels.make_standard_montage('standard_1005') 
    montage_1005_dict = montage_1005.get_positions()
    ch_pos_1005 = montage_1005_dict['ch_pos']

    ch_pos_muse = {}
    for ch in muse_chs:
        ch_pos_muse[ch] = ch_pos_1005[ch]
    montage_muse = mne.channels.make_dig_montage(ch_pos=ch_pos_muse, nasion=montage_1005_dict['nasion'],
                                                  lpa=montage_1005_dict['lpa'], rpa=montage_1005_dict['rpa'],
                                                  hsp=montage_1005_dict['hsp'], hpi=montage_1005_dict['hpi'])
    return montage_muse
    
def makeEvokedArray(data):
    #Code adapted from https://mne.tools/dev/auto_examples/visualization/plot_eeglab_head_sphere.html#sphx-glr-auto-examples-visualization-plot-eeglab-head-sphere-py
    #Takes in a np array of Muse EEG data of 4 channels by # of measurements
    #Returns a mne EvokedArray object

    muse_chs = ['TP9', 'AF7', 'AF8', 'TP10']
    sfreq = 256  # Hz

    info = mne.create_info(muse_chs, sfreq, ch_types='eeg')
    evoked = mne.EvokedArray(data, info)
    montage_muse = makeMuseMontage()

    evoked.set_montage(montage_muse)
    return evoked

def doICA(data):
    # Takes in np array of 4 channels by # of measurements
    # Performs ICA to remove artefacts over 40 uV
    # Returns processed data as np array of same size as input

    #Define ICA object
    seed = 21
    ica = mne.preprocessing.ICA(n_components=4, random_state=seed, method="infomax", max_iter=1000)
    ica.exclude = [0]
    reject = dict(eeg=100e-6) 
    
    #Create mne raw object
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage(makeMuseMontage())
    
    #Fit the ICA to raw, then apply it to a copy of raw
    try:
        ica.fit(raw, reject=reject)
    except:
        return None #No clean segment found

    #Extract np array from mne raw object
    raw_ica = raw.copy()
    ica.apply(raw_ica)
    data_ica = raw_ica.get_data()

    return data_ica

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

def doFFT(channels, fs):
    # Takes in a 2D np array (n points x d channels) and sampling frequency
    # Returns a 2D np array corresponding to the fft and the 1D xf vector
    N = channels.shape[0]
    xf = np.linspace(0.0, fs/2, N//2)
    fft_channels = []
    for channel in channels.T:
        averageAmplitude = np.average(channel)
        channel = channel - averageAmplitude #Remove DC bias
        yf = scipy.fftpack.fft(channel)
        yf = (2/N)*np.abs(yf[:N//2]) #2/N * absolute value and only need the first N/2 data points
        yf[0] = yf[0]*(N/2)
        fft_channels.append(yf)
    return xf, np.stack(fft_channels, axis=-1)

def plotChannelSpectogram(X, Y, Z, ax=None):
    if ax is None:
        ax = plt.gca()
    cm = ax.contourf(X, Y, Z, origin='lower', levels=25, cmap=plt.cm.seismic)
    #plt.imshow(coefficients, origin='lower', extent=[0, len(channel), 0, fs/2], aspect='auto', cmap = 'seismic')
    return cm        

##############################################################################################################################
#Change the openPath, closedPath, and boolean parameters here before running, and comment out breaks at the bottom
cutRecordings = True
cutTime1 = 3.5
cutTime2 = 7
saveFilteredData = False
saveFilteredICAData = False
saveJointPlot = False
saveICAJointPlot = False
saveAnimatedTopomap = False
useICAData = False #set true to try and use the ICA data for FFT and spectogram, better off left false 
saveFFTImage = False
saveFFTCSV = False
saveSpectogramImage = False
saveSpectogramJSONs = False
##############################################################################################################################

openPath = os.getcwd() + r"\\Feb 2021 Raw Data\\OPEN\\"
closedPath = os.getcwd() + r"\\Feb 2021 Raw Data\\CLOSE\\"
openFiles = getFiles(openPath)
closedFiles = getFiles(closedPath)

fs = 256

for lst in [openFiles, closedFiles]:
    for file in lst:
        data_file = loadCSV(file) #4 channels by n points
        name = file.split(r"\\")[-1].split(".")[0]
        print(name, "\n")

        #Cut recordings to cutTime and skip over shorter ones
        data_lst = []
        if cutRecordings:
            if len(data_file) > int(fs*cutTime2):
                data_lst.append(data_file[:int(fs*cutTime1)])
                data_lst.append(data_file[int(fs*cutTime1):int(fs*cutTime2)])
                print("Can be cut into two files")
            elif len(data_file) > int(fs*cutTime1):
                data_lst.append(data_file[:int(fs*cutTime1)])
                print("Can only be cut into one file")
            else:
                print("Not long enough to cut")
                continue
        else:
            data_lst.append(data_file)

        for idx, data in enumerate(data_lst):
            if cutRecordings and len(data_lst) > 1:
                name = file.split(r"\\")[-1].split(".")[0]
                name = name + "_" + str(idx)

            #Filter the data
            data = bandpassFilterChannels(data, 0.5, 40, fs, 5)
            if saveFilteredData:
                np.savetxt(name + " - Filtered.csv", data, delimiter=",")

            #Load data into mne EvokedArray
            data = data.T / 10**6
            evoked = makeEvokedArray(data)

            #Perform ICA on data and make an evoked object for it
            if useICAData:
                data_ica = doICA(data)
                if data_ica is not None:
                    evoked_ica = makeEvokedArray(data_ica)
                    if saveFilteredICAData:
                        np.savetxt(name + " - ICA Filtered.csv", data_ica.T*10**6, delimiter=",")
                
            #Save joint plots
            if saveJointPlot:
                evoked.plot_joint(times='peaks', title=name, show=False, topomap_args={"extrapolate":'box'})
                plt.savefig(name + ".png")
                plt.close()
            if saveICAJointPlot and data_ica is not None:
                evoked_ica.plot_joint(times='peaks', title=name+" - ICA", show=False, topomap_args={"extrapolate":'box'})
                plt.savefig(name + " ICA.png")
                plt.close()        

            #Save animated topomap
            if saveAnimatedTopomap:
                fps = 2
                fig, anim = evoked.animate_topomap(frame_rate=fps, extrapolate='box', show=False, blit=False)
                anim.save(name + ".gif", writer=animation.PillowWriter(fps=fps))
                plt.close()

            #Convert back to n points by 4 channels
            if useICAData and data_ica is not None:
                data = data_ica.T * 10**6
            else:
                data = data.T * 10**6
            
            # Take FFT  
            if saveFFTImage or saveFFTCSV:
                xf, fft_channels = doFFT(data, fs)
                end_idx = 50*int((data.shape[0]//2)/(fs/2)) #only keep up to 50 Hz
                xf = xf[:end_idx]
                fft_channels = fft_channels[:end_idx, :]

            #Save FFT to CSV
            if saveFFTCSV:
                np.savetxt(name + " - FFT.csv", np.concatenate((xf.reshape(-1, 1), fft_channels), axis=1), delimiter=",")
            
            #Plot FFT
            labels = ["Channel 1: TP9", "Channel 2: AF7", "Channel 3: AF8", "Channel 4: TP10"]
            if saveFFTImage:
                num_channels = fft_channels.shape[1]
                colors = ["r", "g", "b", "m"]
                fig, axs = plt.subplots(nrows=num_channels, sharex=True, sharey=True, figsize=(11.0, 8.5))
                for i in range(num_channels):
                    axs[i].plot(xf, fft_channels[:, i], color=colors[i])
                    axs[i].legend([labels[i]])
                    axs[i].set_ylabel(r"Magnitude ($\mu$V)")

                    if i == 0:
                        axs[i].set_title(name + " - FFT", fontweight="bold")
                    if i == num_channels:
                        axs[i].set_xlabel("Frequency (Hz)")
                plt.savefig(name + " - FFT" + ".png")    
                plt.close()

            #Take Morlet Wavelet Transform
            if saveSpectogramImage or saveSpectogramJSONs:
                dt = 1/fs
                
                wavelet_name = "morl"
                scales = pywt.scale2frequency(wavelet=wavelet_name, scale=np.arange(4, 64))*fs #corresponds to 52-2Hz (Scale 64 is 3.3 Hz, 76 is 2.77 Hz, 104 is 2.01 Hz)
                #print(scales)

                if saveSpectogramJSONs:
                    file_spectogram = np.zeros([data.shape[1], len(scales), data.shape[0]]) # 4 channels x len(scales) x timepoints
                    file_spectogram_power = np.zeros([data.shape[1], len(scales), data.shape[0]]) # 4 channels x len(scales) x timepoints

                if saveSpectogramImage:
                    fig, axs = plt.subplots(nrows=data.shape[1], ncols = 2, sharex = True, figsize=(22, 17))
                
                for i, channel in enumerate(data.T):
                    [coefficients, frequencies] = pywt.cwt(channel, scales, wavelet_name, dt)
                    power = (abs(coefficients)) ** 2

                    if saveSpectogramImage:
                        x = np.linspace(0, len(channel)/fs, len(channel))
                        X, Y = np.meshgrid(x, scales)

                        #Plot cwt coefficients and power
                        for j, Z in enumerate([coefficients, power]):
                            #Make subplot
                            scm = plotChannelSpectogram(X, Y, Z, axs[i, j])
                            #Label subplot
                            axs[i, j].text(0.77, 0.86, labels[i], bbox={'facecolor':'white','alpha': 0.5, 'pad': 5}, transform=axs[i, j].transAxes)
                            
                            if i+1 == data.shape[1]:
                                axs[i, j].set_xlabel("Time (s)")
                            axs[i, j].set_ylabel("Pseudo-Frequency (Hz)")
                            fig.colorbar(scm, ax=axs[i, j])

                        #Set titles
                        axs[0, 0].set_title(name + " - Spectogram", fontweight="bold")
                        axs[0, 1].set_title(name + " - Power Spectogram", fontweight="bold") 

                    # Save coefficients, power for each channel
                    if saveSpectogramJSONs:
                        file_spectogram[i, :, :] = coefficients
                        #file_spectogram_power[i, :, :] = power #This can be reprocessed later using saved coefficients

                if saveSpectogramJSONs:
                    with open(name + " - Spec Coefs.json", 'w') as f:
                        json.dump(file_spectogram.tolist(), f)
                    #with open(name + " - Spec Power.json", 'w') as f:
                    #    json.dump(file_spectogram_power.tolist(), f)
                
                if saveSpectogramImage:
                    #plt.show()
                    plt.savefig(name + " - Spectogram" + ".png")
                    plt.close()

        break
    break
print("Complete!")        


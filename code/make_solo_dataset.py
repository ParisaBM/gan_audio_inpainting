import numpy as np
from matplotlib import pyplot as plt
import os
from wavio import read
from scipy.signal import firwin


        
        
def downsample3(sig,Nwin=32):
    win = firwin(numtaps=Nwin, cutoff=0.5)
    new_sig = sig.copy()
    new_sig = np.convolve(new_sig,win, 'same')
    new_sig = new_sig[2::3]
    return new_sig

def normalize(x):
    m = np.max(np.abs(x))
    return x/m

def toint16(x):
    return np.int16(x*(2**15))

def tomono(x):
    return (x[:,0]+x[:,1])/2

if __name__ == '__main__':

    pathaudio = '../data/solo'

    paths = []
    for root, dirs, files in os.walk(pathaudio):
        for file in files:
            if file[-3:]=='wav':
                paths.append(os.path.join(root, file))


    waveforms = np.array([], dtype=np.int16)
    for i, filepath in enumerate(paths):
        wavobj = read(paths[i])
        fs = wavobj.rate
        assert(fs==44100)
        waveform = wavobj.data.copy()
        waveform = normalize(waveform)
        waveform = waveform[:, 0]
        waveform = downsample3(waveform)
        waveform = toint16(waveform)
        waveforms = np.append(waveforms, waveform, axis=0)

    np.savez('../data/waveforms', waveforms)

    pathaudio = '../data/noise'

    paths = []
    for root, dirs, files in os.walk(pathaudio):
        for file in files:
            if file[-3:]=='wav':
                paths.append(os.path.join(root, file))

    noise_len = 4 * 1024

    waveforms = np.empty((0, noise_len))
    for i, filepath in enumerate(paths):
        wavobj = read(paths[i])
        fs = wavobj.rate
        assert(fs==44100)
        waveform = wavobj.data.copy()
        waveform = normalize(waveform)
        waveform = waveform[:, 0]
        waveform = downsample3(waveform)
        waveform = toint16(waveform)
        waveform = waveform[:len(waveform) // noise_len * noise_len]
        waveform = np.reshape(waveform, (-1, noise_len))
        waveforms = np.concatenate((waveforms, waveform))
    
    np.random.shuffle(waveforms)
    np.savez('../data/noise', np.reshape(waveforms, (-1,)))
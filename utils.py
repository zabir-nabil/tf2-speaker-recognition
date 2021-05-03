import os
import numpy as np
import librosa
import numpy as np
import time as timelib
import scipy
import soundfile as sf
import scipy.signal as sps
from scipy import interpolate


# GPU Setup

def test_GPU(args):
    # Initialize GPUs
    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Data loading

def load_wav_fast(vid_path, sr, mode='train'):
    """load_wav() is really slow on this version of librosa.
    load_wav_fast() is faster but we are not ensuring a consistent sampling rate"""
    wav, sr_ret = sf.read(vid_path)

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav

def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)

def get_chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]


def debug_generator(generator):
    import cv2
    import pdb
    G = generator.next()
    for i,img in enumerate(G[0]):
        path = '../sample/{}.jpg'.format(i)
        img = np.asarray(img[:,:,::-1] + 128.0, dtype='uint8')
        cv2.imwrite(path, img)


# set up multiprocessing
def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool


# vggface2 dataset
def get_vggface2_imglist(args):
    def get_datalist(s):
        file = open('{}'.format(s), 'r')
        datalist = file.readlines()
        imglist = []
        labellist = []
        for i in datalist:
            linesplit = i.split(' ')
            imglist.append(linesplit[0])
            labellist.append(int(linesplit[1][:-1]))
        return imglist, labellist

    print('==> calculating image lists...')
    # Prepare training data.
    imgs_list_trn, lbs_list_trn = get_datalist(args.trn_meta)
    imgs_list_trn = [os.path.join(args.data_path, i) for i in imgs_list_trn]
    imgs_list_trn = np.array(imgs_list_trn)
    lbs_list_trn = np.array(lbs_list_trn)

    # Prepare validation data.
    imgs_list_val, lbs_list_val = get_datalist(args.val_meta)
    imgs_list_val = [os.path.join(args.data_path, i) for i in imgs_list_val]
    imgs_list_val = np.array(imgs_list_val)
    lbs_list_val = np.array(lbs_list_val)

    return imgs_list_trn, lbs_list_trn, imgs_list_val, lbs_list_val


def get_imagenet_imglist(args, trn_meta_path='', val_meta_path=''):
    with open(trn_meta_path) as f:
        strings = f.readlines()
        trn_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        trn_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()

    with open(val_meta_path) as f:
        strings = f.readlines()
        val_list = np.array([os.path.join(args.data_path, '/'.join(string.split()[0].split(os.sep)[-4:]))
                             for string in strings])
        val_lb = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return trn_list, trn_lb, val_list, val_lb


def get_voxceleb2_datalist(args, path):
    with open(path) as f:
        strings = f.readlines()
        audiolist = np.array([os.path.join(args.data_path, string.split()[0]) for string in strings])
        labellist = np.array([int(string.split()[1]) for string in strings])
        f.close()
    return audiolist, labellist

def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh


def sync_model(src_model, tgt_model):
    print('==> synchronizing the model weights.')
    params = {}
    for l in src_model.layers:
        params['{}'.format(l.name)] = l.get_weights()

    for l in tgt_model.layers:
        if len(l.get_weights()) > 0:
            l.set_weights(params['{}'.format(l.name)])
    return tgt_model
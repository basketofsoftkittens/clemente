"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains utilities for audio processing as well as
pre-processing scripts. This script will be executed to extract features for
training.
"""

import os
import os.path as path
import sys
import h5py
import numpy as np
import scipy.signal
from tqdm import tqdm
import random
import wave

SAMPLE_RATE = 44100
FFT_SIZE = 2048
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH = 1024
WIN_LENGTH = 2048

# dir
DATA_DIR = path.join(".", "data")
AUDIO_DIR = path.join(DATA_DIR, "wav")
BIN_DIR = path.join(DATA_DIR, "bin")


def get_spectrograms(sound_file, sr=SAMPLE_RATE, fft_size=FFT_SIZE):
    """Resamples audio file to sample rate defined by sr, then obtains short-
    time Fourier transform. This matrix is transposed so time is in the first
    axis and then returned.

    TODO: Implement the resampling, STFT, and related logic required

    Args:
        sound_file (str): filepath of audio file to extract spectrogram for
        sr (int): sample rate the file will be resampled to before stft
        fft_size (int): size of FFT

    Returns:
        np.float32 magnitude matrix of shape (T, 1+n_fft/2)
    """
    # open the wav file
    with wave.open(sound_file) as f:
        original_sample_rate = f.getframerate()
        wave_data = np.array(list(f.readframes(f.getnframes())))
    # resample if necessary
    if original_sample_rate != sr:
        up = sr
        down = original_sample_rate
        if sr < original_sample_rate:
            up = original_sample_rate
            down = sr
        wave_data = scipy.signal.resample_poly(wave_data, window='hann', up=up, down=down)
    # stft
    transformed_wave_data = scipy.signal.stft(
        wave_data, window="hann", fs=sr, nperseg=WIN_LENGTH, noverlap=HOP_LENGTH, nfft=fft_size
    )[-1]
    return transformed_wave_data.T


def read_list(listfile):
    """Reads a text file and puts each line into a list item

    TODO: Implement this function

    Args:
        listfile (str): path to mos_list.txt file

    Returns:
        moslist (list of str): list of lines in file as string
            ex: [ "a.wav,2.53", "b.wav,4.6", ... ]
    """
    with open(listfile) as f:
        moslist = [line.strip() for line in f.readlines()]
    return moslist


def read_bin(file_path):
    """Read in spectrogram from h5 binary filepath

    TODO: Implement this function

    Args:
        file_path (str): path to hdf5 binary file

    Returns:
        mag_sgram of shape (1, t, SGRAM_DIM) in dict with key 'mag_sgram'
    """
    with h5py.File(file_path, 'r') as f:
        ds = np.expand_dims(f["mag_sgram"][()], 0)
    return {"mag_sgram": ds}


def pad(array, reference_shape):
    """Pad with zeros to fit array to reference_shape

    TODO: Nothing
    """
    result = np.zeros(reference_shape)
    result[: array.shape[0], : array.shape[1], : array.shape[2]] = array
    return result


def data_generator(file_list, bin_root, frame=True, batch_size=1):
    """This function is the generator function called by fit(), which returns
    feature arrays for training

    TODO: Nothing, unless you have any ideas for improvement!

    Args:
        file_list (list of "filepath,mos" strings): contains all files to have
            data extracted and their corresponding MOS values
        bin_root (str): binary file directory
        frame (bool): Determines whether or not to return frame-wise score
    """
    index = 0
    while True:
        # Build list of filenames of file_list, omitting ext, up to batch_size
        filename = [
            file_list[index + x].split(",")[0].split(".")[0] for x in range(batch_size)
        ]

        for i in range(len(filename)):
            # for each filename in batch list
            all_feat = read_bin(path.join(bin_root, filename[i] + ".h5"))
            sgram = all_feat["mag_sgram"]

            # the very first feat
            if i == 0:
                feat = sgram
                max_timestep = feat.shape[1]
            else:
                if sgram.shape[1] > feat.shape[1]:
                    # extend all feat in feat
                    ref_shape = [feat.shape[0], sgram.shape[1], feat.shape[2]]
                    feat = pad(feat, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                elif sgram.shape[1] < feat.shape[1]:
                    # extend sgram to feat.shape[1]
                    ref_shape = [sgram.shape[0], feat.shape[1], feat.shape[2]]
                    sgram = pad(sgram, ref_shape)
                    feat = np.append(feat, sgram, axis=0)
                else:
                    # same timestep, append all
                    feat = np.append(feat, sgram, axis=0)

        mos = [float(file_list[x + index].split(",")[1]) for x in range(batch_size)]
        mos = np.asarray(mos).reshape([batch_size])
        frame_mos = np.array(
            [mos[i] * np.ones([feat.shape[1], 1]) for i in range(batch_size)]
        )

        index += batch_size
        # ensure next batch won't out of range
        if index + batch_size >= len(file_list):
            index = 0
            random.shuffle(file_list)
        if frame:
            yield feat, (mos, frame_mos)
        else:
            yield feat, (mos)


def extract_to_h5(audio_dir, bin_dir):
    """For each wav file in ./data/wav, extract spectrogram, and save data in
    .h5 file in ./data/bin. Matrix will be saved under 'mag_sgram' key.

    TODO: Nothing

    Args:
        audio_dir (str): audio file directory
        bin_dir (str): binary file directory
    """
    print("audio dir: {}".format(path.normpath(audio_dir)))
    print("bin_dir: {}".format(path.normpath(bin_dir)))

    if not path.exists(bin_dir):
        os.makedirs(bin_dir)
    if len(os.listdir(bin_dir)) != 0:
        for file in os.listdir(bin_dir):
            os.remove(path.join(bin_dir, file))

    # get filenames
    files = []
    for f in os.listdir(audio_dir):
        if f.endswith(".wav"):
            files.append(f.split(".")[0])

    for i in tqdm(range(len(files))):
        f = files[i]

        # set audio/visual file path
        audio_file = path.join(audio_dir, f + ".wav")

        # spectrogram
        mag = get_spectrograms(audio_file)

        with h5py.File(path.join(bin_dir, "{}.h5".format(f)), "w") as hf:
            hf.create_dataset("mag_sgram", data=mag)


def extract_features():
    extract_to_h5(AUDIO_DIR, BIN_DIR)


if __name__ == "__main__":
    extract_features()

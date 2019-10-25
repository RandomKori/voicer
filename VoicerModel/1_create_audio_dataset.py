import pandas as pd
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
from processing.proc_audio import get_padded_spectros
from hparams import *
import tensorflow as tf
sess = tf.Session()

print('Loading the data...')
metadata = pd.read_csv('data/voice/Ruslan/metadata.csv',
                       dtype='object', quoting=3, sep='|', header=None)
# uncomment this line if you yave weak GPU
# metadata = metadata.iloc[:500]

# audio filenames
dot_wav_filenames = metadata[0].values


print('Processing the audio samples (computation of spectrograms)...')
for filename in tqdm(dot_wav_filenames):
    file_path = 'data/voice/Ruslan/' + filename + '.wav'
    fname, mel_spectro, spectro = get_padded_spectros(file_path, r,
                                                      PREEMPHASIS, N_FFT,
                                                      HOP_LENGTH, WIN_LENGTH,
                                                      SAMPLING_RATE,
                                                      N_MEL, REF_DB,
                                                      MAX_DB)

    decod_inp_tensor = tf.concat((tf.zeros_like(mel_spectro[:1, :]),
                                  mel_spectro[:-1, :]), 0)
    decod_inp = sess.run(decod_inp_tensor)
    decod_inp = decod_inp[:, -N_MEL:]

    # Padding of the temporal dimension
    dim0_mel_spectro = mel_spectro.shape[0]
    dim1_mel_spectro = mel_spectro.shape[1]
    padded_mel_spectro = np.zeros((MAX_MEL_TIME_LENGTH, dim1_mel_spectro))
    padded_mel_spectro[:dim0_mel_spectro, :dim1_mel_spectro] = mel_spectro

    dim0_decod_inp = decod_inp.shape[0]
    dim1_decod_inp = decod_inp.shape[1]
    padded_decod_input = np.zeros((MAX_MEL_TIME_LENGTH, dim1_decod_inp))
    padded_decod_input[:dim0_decod_inp, :dim1_decod_inp] = decod_inp

    dim0_spectro = spectro.shape[0]
    dim1_spectro = spectro.shape[1]
    padded_spectro = np.zeros((MAX_MAG_TIME_LENGTH, dim1_spectro))
    padded_spectro[:dim0_spectro, :dim1_spectro] = spectro

    joblib.dump(padded_mel_spectro,"data/mel_spectro_training_"+filename+".pkl")
    joblib.dump(padded_spectro,"data/spectro_training_"+filename+".pkl")
    joblib.dump(padded_decod_input,"data/decoder_input_training_"+filename+".pkl")

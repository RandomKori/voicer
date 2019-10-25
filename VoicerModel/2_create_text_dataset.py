import pandas as pd
from sklearn.externals import joblib
from processing.proc_text import transform_text_for_ml
from hparams import *
from tqdm import tqdm

# import data
metadata = pd.read_csv('data/voice/Ruslan/metadata.csv',
                       dtype='object', quoting=3, sep='|',
                       header=None)

# uncomment this line if you yave weak GPU
metadata = metadata.iloc[:3000]

metadata['norm_lower'] = metadata[2].apply(lambda x: x.lower())
texts = metadata['norm_lower']

# Infer the vocabulary
vocabulary = ' !"”„“«»‑#$%&\'()*+,-–./0123456789:;<=>?@abcdefghijklmnopqrstuvwxyz[]^_…`’\'́\\{|}~абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
vocabulary += 'P'  # add padding character

print(vocabulary)
# Create association between vocabulary and id
vocabulary_id = {}
i = 0
for char in list(vocabulary):
    vocabulary_id[char] = i
    i += 1


text_input_ml = transform_text_for_ml(texts.values,
                                      vocabulary_id,
                                      NB_CHARS_MAX)

print (text_input_ml)
print (vocabulary)
# split into training and testing
#len_train = int(TRAIN_SET_RATIO * len(metadata))
#text_input_ml_training = text_input_ml[:len_train]
#text_input_ml_testing = text_input_ml[len_train:]

# save data
#joblib.dump(text_input_ml_training, 'data/text_input_ml_training.pkl')
#joblib.dump(text_input_ml_testing, 'data/text_input_ml_testing.pkl')
dot_wav_filenames = metadata[0].values
i = 0
for filename in tqdm(dot_wav_filenames):
    joblib.dump(text_input_ml[i], 'data/text_input_ml_training_'+filename+'.pkl')
    i += i


joblib.dump(vocabulary_id, 'data/vocabulary.pkl')

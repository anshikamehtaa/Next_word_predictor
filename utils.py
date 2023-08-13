import pickle
import pandas as pd
import requests
import tensorflow as tf
import numpy as np

df = pd.read_csv('tmdb_5000_movies.csv')
df = df['original_title']

movie_name = df.to_list()

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(movie_name)
seq = tokenizer.texts_to_sequences(movie_name)


def make_prediction(text, n_words):
    model = pickle.load(open("model.pkl",'rb'))
    vocab_array = np.array(list(tokenizer.word_index.keys()))
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        print(vocab_array[np.argsort(model.predict(text_padded)) - 1].ravel()[:-3])
        text += " " + prediction
    return text

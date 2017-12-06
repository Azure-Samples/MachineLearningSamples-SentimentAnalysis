# Sentiment Analysis of Movie Reviews using Deep Learning

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing import sequence
from keras.datasets import imdb
from azureml.dataprep.package import run
from azureml.logging import get_azureml_logger
import h5py
import numpy as np
import pandas as pd
import csv
import argparse

# initialize the logger

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.SentimentAnalysis.SentimentExtraction','true')

def read_reviews_from_csv(dataset):
    '''
    Reads the csv file containing reviews and sentiments.
    @param
        dataset = input dataset
    @returns:
        df:       a dataframe containing the reviews and sentiments
    '''
    df = pd.read_csv(dataset, encoding='cp437', sep='|')
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df

def train_model(ratio=.5):
    '''
    Main function to build the model. The funcion sets parameters for building the model.
    @returns:
        model:       model built using the reviews
    '''
    # set parameters:
    max_features = 5000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 2
    seed = 113
    # get the reviews_list and labels_ist from the csv file
    df = run('sampleReviews.dprep', dataflow_idx=0)

    rows, columns = df.shape
    reviews_list = []
    labels_list = []
    
    for i in range(0, rows):
        try:
            labels_list.append(int(float(df.iloc[i,1])))
            reviews_list.append(df.iloc[i,0])
        except UnicodeEncodeError:
            pass

    # get the corresponding vectors from the data set
    reviews_list_vec = get_vectors_from_text(reviews_list)
    # shuffle the data set
    np.random.seed(seed)
    np.random.shuffle(reviews_list_vec)
    np.random.seed(seed)
    np.random.shuffle(labels_list)
    # split the data set into train and test data
    x_train = reviews_list_vec[:int(len(reviews_list)*ratio)]
    y_train = labels_list[:int(len(labels_list)*ratio)]
    x_test = reviews_list_vec[int(len(reviews_list)*ratio):]
    y_test = labels_list[int(len(labels_list)*ratio):]
    print('Building model...')
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    return model

def get_vectors_from_text(dataset_list,word_to_ind=imdb.get_word_index(),
                           start_char=1,
                           index_from=3,
                           maxlen=400,
                           num_words=5000,
                          oov_char=2,skip_top=0):
    '''
    Gets the list vector mapped according to the word to indices dictionary.
    
    @param
        dataset_list = list of review texts in unicode format
        word_to_ind = word to indices dictionary
        hyperparameters: start_char-->sentence starting after this char.
                        index_from-->indices below this will not be encoded.
                        max-len-->maximum length of the sequence to be considered.
                        num_words-->number of words to be considered according to the rank.Rank is
                                    given according to the frequency of occurence
                        oov_char-->out of variable character.
                        skip_top-->no of top rank words to be skipped
    @returns:
        x_train:       Final list of vectors(as list) of the review texts
    '''
    x_train = []
    for review_string in dataset_list:
        review_string_list = text_to_word_sequence(review_string)
        review_string_list = [ele for ele in review_string_list]
        
        x_predict = []
        for i in range(len(review_string_list)):
            if review_string_list[i] not in word_to_ind:
                continue
            x_predict.append(word_to_ind[review_string_list[i]])
        x_train.append((x_predict))
    # add te start char and also take care of indexfrom
    if start_char is not None:
        x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
    elif index_from:
        x_train = [[w + index_from for w in x] for x in x_train]
    # only maxlen is out criteria
    x_train=[ele[:maxlen] for ele in x_train]
    # if num is not given take care
    if not num_words:
        num_words = max([max(x) for x in x_train])
    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        x_train = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in x_train]
    else:
        x_train = [[w for w in x if (skip_top <= w < num_words)] for x in x_train]
    # padd the sequences
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # return the vectors form of the text
    return x_train

def predict_review(model,review_text):
    '''
    Predict the sentiment of the review text.

    @param
        model:       SequentialModel which we trained the data on.
        review_text:        Review text to be predicted on
    @returns
        sentiment score on the review text.
    '''
    # convert the review text into vector 
    x_predict = get_vectors_from_text([review_text])[0]
    # reshape the x_predict 
    x_predict = np.reshape(x_predict,(1,len(x_predict)))
    # predict on the model
    return model.predict(x_predict)[0][0]

# the dataset in the csv format
dataset = 'sampleReviews.txt'
review_text = 'i loved the movie'

# now train the model using the dataset
model = train_model()
print("Review Sentiment:", predict_review(model, review_text.lower()))
model.save('./outputs/sentModel.h5')

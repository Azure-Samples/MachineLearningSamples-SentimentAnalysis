# This script generates the scoring and schema files necessary to opearaitonalize Sentiment Analysis
# Init and run functions
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
import pandas
import numpy as np
import os

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

# init loads the model (global)
def init():
    from sklearn.externals import joblib
    from keras.models import Sequential
    from keras.models import load_model
    import h5py

    # load the model file
    global model
    model = load_model('sentModel.h5')

# run takes an input dataframe and performs sentiment prediction
def run(input_df):
    import json
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers import Embedding
    from keras.layers import Conv1D, GlobalMaxPooling1D
    from keras.preprocessing.text import text_to_word_sequence
    from keras.preprocessing import sequence
    from keras.datasets import imdb
    import h5py    

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

    pred = predict_review(model, input_df['reviewText'][0])
    return json.dumps(str(pred))

def main():
  from azureml.api.schema.dataTypes import DataTypes
  from azureml.api.schema.sampleDefinition import SampleDefinition
  from azureml.api.realtime.services import generate_schema
  import pandas
  
  # create the outputs folder
  os.makedirs('./outputs', exist_ok=True)

  df = pandas.DataFrame(data=[['i loved the new movie and enjoyed the great acting']], columns=['reviewText'])

  # Turn on data collection debug mode to view output in stdout
  os.environ["AML_MODEL_DC_DEBUG"] = 'true'

  # Test the output of the functions
  init()
  input1 = pandas.DataFrame(data=[['i loved the new movie and enjoyed the great acting']], columns=['reviewText'])

  print("Result: " + run(input1))
  
  inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}
  
  #Genereate the schema
  generate_schema(run_func=run, inputs=inputs, filepath='myschema.json')
  print("Schema generated")

if __name__ == '__main__':
    #init()
    #input1 = pandas.DataFrame(data=[['i loved the new movie and enjoyed the great acting']], columns=['reviewText'])
    #res = run(input1)
    #print(res)
    main()

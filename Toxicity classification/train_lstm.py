import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import argparse, random

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint

import gensim.models.keyedvectors as word2vec
import gc


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default = './saved_models', help='path for saving checkpoints')
parser.add_argument('--results_dir', type=str, default = './results', help='path for saving results')
parser.add_argument('--data_dir', default = './data', help='data directory')

parser.add_argument('--word_vector', default = 'word2vec', help='word vector for training')
parser.add_argument('--train',action='store_true')
parser.add_argument('--resume',action='store_true')

parser.add_argument('--num_hidden', type=float, default=60)
parser.add_argument('--learning_rate', type=float, default=0.00002)
parser.add_argument('--adam_epsilon', type=float, default=0.00000001)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--num_save_epochs', type=int, default=1)

args = parser.parse_args()


"""Setting seeds for reproducibility"""

random.seed(args.seed)
np.random.seed(args.seed)


"""Download the train and test datasets, either from kaggle or from the following links, and place them in args.data_dir folder"""

# train.csv :                   "https://drive.google.com/file/d/1--TceffCWOdmOv_oq-ryHn9NDifAwsTo/view?usp=sharing"
# test_public_expanded.csv:     "https://drive.google.com/file/d/1-8yHngJrWfS_cirwbXY7kUYqxkWpRrxB/view?usp=sharing"

# Kaggle link: "https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data"


"""Also, download and unzip the word vector files and place them in args.data_dir folder"""

# word2vec :                    "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
# hard-debiased-word2vec :      "https://drive.google.com/file/d/0B5vZVlu2WoS5ZTBSekpUX0RSNDg/view?usp=sharing"
# GloVe :                       "https://drive.google.com/file/d/1jrbQmpB5ZNH4w54yujeAvNFAfVEG0SuE/view?usp=sharing"
# GN-Glove :                    "https://drive.google.com/file/d/1v82WF43w-lE-vpZd0JC1K8WYZQkTy_ii/view?usp=sharing"


""" Downsampling the data """


train_raw = pd.read_csv(os.path.join(args.data_dir,'train.csv'))
# print(train_raw.shape)
# print(train_raw.head(10))

test = pd.read_csv(os.path.join(args.data_dir,'test_public_expanded.csv'))
# print(test.shape)
# print(test.head(10))

train_raw['comment_text'] = train_raw['comment_text'].astype(str)
test['comment_text'] = test['comment_text'].astype(str)

train_raw=train_raw.fillna(0)
test = test.fillna(0)

# convert target to 0,1
train_raw['target']=(train_raw['target']>=0.5).astype(float)
test['target']=(test['toxicity']>=0.5).astype(float)

raw_labels = train_raw['target'].values.flatten()
pos_raw = np.where(raw_labels == 1.0)[0]
neg_raw = np.where(raw_labels == 0.0)[0]
white_idx = train_raw[train_raw['white'] >= 0.5].index.values
black_idx = train_raw[train_raw['black'] >= 0.5].index.values
nidx = np.random.choice(neg_raw, len(pos_raw))
pidx = np.random.choice(pos_raw, len(pos_raw))
idxs = np.unique(np.concatenate((nidx, pidx, white_idx, black_idx), axis = 0))

# train = train_raw.iloc[idxs]
train = train_raw  

print(f"{len(train[train['target'] == 1.0 ])} pos; {len(train[train['target'] == 0.0])} neg")

y = train['target']
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]


"""Train tokenizer and make embedding matrix from word vectors"""

max_features = 20000

print("Now fitting tokenizer on the words.")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)


X_t = pad_sequences(list_tokenized_train, maxlen=args.max_length)
X_te = pad_sequences(list_tokenized_test, maxlen=args.max_length)


"""Define and initialize the embedding matrix from the chosen args.word_vector file"""

def loadEmbeddingMatrix(typeToLoad):
        print("Now loading the embeddings matrix for the "+typeToLoad+" word vectors.")

        if(typeToLoad=="glove"):
            EMBEDDING_FILE= os.path.join(args.data_dir, "vectors.txt")
        elif(typeToLoad=="gn-glove"):
            EMBEDDING_FILE= os.path.join(args.data_dir, "vectors300.txt")
        elif(typeToLoad=="word2vec"):
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format(os.path.join(args.data_dir, "GoogleNews-vectors-negative300.bin"), binary=True)
            embed_size = 300
        elif(typeToLoad=="hd-word2vec"):
            word2vecDict = word2vec.KeyedVectors.load_word2vec_format(os.path.join(args.data_dir, "GoogleNews-vectors-negative300-hard-debiased.bin") , binary=True)
            embed_size = 300

        if(typeToLoad=="glove" or typeToLoad=="gn-glove" ):
            embeddings_index = dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f = open(EMBEDDING_FILE)
            for line in f:
                #split up line into an indexed array
                values = line.split()
                #first index is word
                word = values[0]
                #store the rest of the values in the array as a new array
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))
        else:
            embeddings_index = dict()
            for word in word2vecDict.wv.vocab:
                embeddings_index[word] = word2vecDict.word_vec(word)
            print('Loaded %s word vectors.' % len(embeddings_index))
            
        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the same statistics for the rest of our own random generated weights. 
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        
        nb_words = len(tokenizer.word_index)
        #We are going to set the embedding size to the pretrained dimension as we are replicating it. The size will be Number of Words in Vocab X Embedding Size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()

        #With the newly created embedding matrix, we'll fill it up with the words that we have in both our own dictionary and loaded pretrained embedding. 
        embeddedCount = 0
        for word, i in tokenizer.word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
        
        del(embeddings_index)
        gc.collect()
        
        #finally, return the embedding matrix
        return embedding_matrix



embedding_matrix = loadEmbeddingMatrix(args.word_vector)

print(embedding_matrix.shape)



"""Defining the BiLSTM classifier architecture"""

inp = Input(shape=(args.max_length, )) 

x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)

x = Bidirectional(LSTM(args.num_hidden, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Defined model summary:")
model.summary()



"""Run the training with the provided hyperparams"""

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if(args.train):
    print("Now training the model")
    checkpoint = ModelCheckpoint(os.path.join(args.save_dir, "best_model.hdf5"), monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
    epoch_checkpoint = ModelCheckpoint(os.path.join(args.save_dir, "saved-model-{epoch:02d}-{val_accuracy:.2f}.hdf5"), monitor='val_accuracy', verbose=0, save_best_only=False, mode='auto')


    hist = model.fit(X_t,y, batch_size=args.batch_size, epochs=args.num_epochs , validation_split=0.1, verbose = 1, callbacks=[checkpoint, epoch_checkpoint])



"""Getting and saving predictions and embeddings"""
model.load_weights(os.path.join(args.save_dir, "best_model.hdf5"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Loaded model weights from file")

y_predict = model.predict(X_te, verbose = 1)

test['predictions']=(y_predict.flatten()>=0.5).astype(float)

if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

test.to_csv(os.path.join(args.results_dir,'test_predictions.csv'))

embedding_layer_model = Model(inputs=model.input, outputs=model.get_layer('dropout_1').output)

sen_embeddings = embedding_layer_model.predict(X_te, verbose = 1)

sen_embeddings = np.array(sen_embeddings)

with open(os.path.join(args.results_dir, 'second2last.npy'), 'wb') as f:
    np.save(f, sen_embeddings)
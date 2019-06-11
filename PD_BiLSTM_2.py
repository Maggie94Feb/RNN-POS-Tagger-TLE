# -*- coding: utf-8 -*-


import json
import keras
from datetime import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional, Input, Masking, TimeDistributed, Flatten
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

def load(datapath):
    f = open(datapath, 'r')
    data = json.load(f)
    return data['dataset'], data['labels'], dict(data['word_index'])


class revivification:
    def __init__(self, dataset, word_index):
        self.dataset = dataset
        self.word_index = word_index
        self.corpus = []

    def reStore(self):
        for datum in self.dataset:
            sentence = ''.join(list(map(lambda wordindex: next((k for k, v in self.word_index.items(
            ) if v == wordindex), None), list(filter(lambda wordindex: wordindex != 0, datum)))))
            self.corpus.append(sentence)
        return self.corpus


class nn_simple:
    def __init__(self, dataset, labels, correct_labels, dev_dataset, dev_labels, correct_labels_dev, wordvocab):
        self.dataset = np.array(dataset)
        self.labels = np.array(labels)
        self.wordvocab = wordvocab
        self.dev_dataset = dev_dataset
        self.dev_labels = dev_labels
        self.correct_labels = correct_labels
        self.correct_labels_dev = correct_labels_dev

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 32  # the vector size a word need to be converted
        maxlen = 30  # the size of a sentence vector
        outputDims = 18 + 1
        hiddenDims = 32#100
        batchSize = 32
        EPOCH = 5
        train_X = self.dataset
        train_Y = np_utils.to_categorical(self.labels, outputDims)
        dev_X = self.dev_dataset
        dev_Y = np_utils.to_categorical(self.dev_labels, outputDims)
        correct_labels = self.correct_labels
        correct_labels_dev = self.correct_labels_dev

        print(train_X.shape)
        print(train_Y.shape)
        max_features = vocabSize + 1
        word_input = Input(shape=(maxlen,), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(max_features, embeddingDim,
                             input_length=maxlen, name='word_emb')(mask)
        bilstm1 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(word_emb)
        output = TimeDistributed(
            Dense(outputDims, activation='softmax'))(bilstm1)
        model = Model(inputs=[word_input], outputs=output)
        #sgd = optimizers.SGD(lr=0.1, decay=1e-3)
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'], )
        result = model.fit(train_X, train_Y, batch_size=batchSize, epochs=EPOCH, verbose=1,
              validation_data=(dev_X, dev_Y))
        model.save('PDmodel_epoch_10_batchsize_32_embeddingDim_100_new2.h5')
        
        return result

    def save2json(self, json_string, savepath):
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(json_string)
        return "save done."


class nn:
    def __init__(self, dataset, labels, correct_labels, dev_dataset, dev_labels, correct_labels_dev, wordvocab):
        self.dataset = np.array(dataset)
        self.labels = np.array(labels)
        self.wordvocab = wordvocab
        self.dev_dataset = dev_dataset
        self.dev_labels = dev_labels
        self.correct_labels = correct_labels
        self.correct_labels_dev = correct_labels_dev

    def trainingModel(self):
        vocabSize = len(self.wordvocab)
        embeddingDim = 32  # the vector size a word need to be converted
        maxlen = 30  # the size of a sentence vector
        outputDims = 18 + 1
        hiddenDims = 100
        batchSize = 32
        EPOCH = 5
        train_X = self.dataset
        train_Y = np_utils.to_categorical(self.labels, outputDims)
        dev_X = self.dev_dataset
        dev_Y = np_utils.to_categorical(self.dev_labels, outputDims)
        correct_labels = self.correct_labels
        correct_labels_dev = self.correct_labels_dev

        print(train_X.shape)
        print(train_Y.shape)
        max_features = vocabSize + 1
        word_input = Input(shape=(maxlen,), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(max_features, embeddingDim,
                             input_length=maxlen, name='word_emb')(mask)
        bilstm1 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(word_emb)
#        bilstm2 = Bidirectional(
#            LSTM(hiddenDims, return_sequences=True))(bilstm1)
        bilstm_d = Dropout(0.5)(bilstm1)
        output = TimeDistributed(
            Dense(outputDims, activation='softmax'))(bilstm_d)
        flat = Flatten()(output)
        drop = Dropout(0.2)(flat)
        output2 = Dense(1, activation='relu', name='main_output')(drop)
        model = Model(inputs=[word_input], outputs=[output,output2])
        #sgd = optimizers.SGD(lr=0.1, decay=1e-3)
        model.summary()
        model.compile(loss=['categorical_crossentropy','binary_crossentropy'],
                      optimizer='adam',metrics=['accuracy'], loss_weights=[0.5, 0.5])
        model_tagger = Model(inputs=[word_input], outputs=output)
        model_tagger.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        result = model.fit(train_X, [train_Y,correct_labels], batch_size=batchSize, epochs=EPOCH, verbose=1,
              validation_data=(dev_X, [dev_Y, correct_labels_dev]))
#        model.save('PDmodel_epoch_10_batchsize_32_embeddingDim_100_new2.h5')
        model_tagger.save('PDmodel_epoch_10_batchsize_32_embeddingDim_100_new2.h5')
        
        return result

    def save2json(self, json_string, savepath):
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(json_string)
        return "save done."




if __name__ == '__main__':
    ts = dt.now()
    te = dt.now()
    spent = te - ts
    print('[Finished in %s]' % spent)

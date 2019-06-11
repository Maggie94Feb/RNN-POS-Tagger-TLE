# -*- coding: utf-8 -*-

import data_loader
from PD_BiLSTM_2 import *
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import np_utils
import pickle
import os
import numpy as np
# preprocess learner corpus
meta_list, data_list = data_loader.load_data(load_train=True, load_dev=True, load_test=True)
train_meta, train_meta_corrected, \
dev_meta, dev_meta_corrected, \
test_meta, test_meta_corrected = meta_list

train_data, train_data_corrected, \
dev_data, dev_data_corrected, \
test_data, test_data_corrected = data_list

# process corrected and incorrect data
X_train = [[d["form"].tolist(), d["upostag"].tolist()] for d in train_data]
X_train_correct = [[d["form"].tolist(), d["upostag"].tolist()] for d in train_data_corrected]
X_train.extend(X_train_correct)
correct_labels = np.array([i<4124 and 0 or 1 for i in range(len(X_train))])
X_dev = [[d["form"].tolist(), d["upostag"].tolist()] for d in dev_data]
X_dev_correct = [[d["form"].tolist(), d["upostag"].tolist()] for d in dev_data_corrected]
X_dev.extend(X_dev_correct)
correct_labels_dev = np.array([i<500 and 0 or 1 for i in range(len(X_dev))])
X_test = [[d["form"].tolist(), d["upostag"].tolist()] for d in test_data]
X_test_correct = [[d["form"].tolist(), d["upostag"].tolist()] for d in test_data_corrected]
X_test.extend(X_test_correct)

# generate worddict和tagdict
word_to_ix = dict()
tag_to_ix = dict()
for i in range(len(X_train)):
    sent = X_train[i][0]
    tags = X_train[i][1]
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
if "_" not in word_to_ix:
    word_to_ix["_"] = len(word_to_ix.keys())
if "_" not in tag_to_ix:
    tag_to_ix["_"] = len(tag_to_ix.keys())
    
MAXLEN=30
train_X,train_y = [],[]
dev_X,dev_y = [],[]
test_X,test_y = [],[]

train_X = [ [word_to_ix.get(w,0) for w in item[0]] for item in X_train]
train_X = pad_sequences(train_X, maxlen=MAXLEN, padding='post', truncating='post', value=0)

dev_X = [ [word_to_ix.get(w,0) for w in item[0]] for item in X_dev]
dev_X = pad_sequences(dev_X, maxlen=MAXLEN, padding='post', truncating='post', value=0)

test_X = [ [word_to_ix.get(w,0) for w in item[0]] for item in X_test]
test_X = pad_sequences(test_X, maxlen=MAXLEN, padding='post', truncating='post', value=0)

train_y = [ [tag_to_ix.get(w,0) for w in item[1]] for item in X_train]
train_y = pad_sequences(train_y, maxlen=MAXLEN, padding='post', truncating='post', value=0)

dev_y = [ [tag_to_ix.get(w,0) for w in item[1]] for item in X_dev]
dev_y = pad_sequences(dev_y, maxlen=MAXLEN, padding='post', truncating='post', value=0)

test_y = [ [tag_to_ix.get(w,0) for w in item[1]] for item in X_test]
test_y = pad_sequences(test_y, maxlen=MAXLEN, padding='post', truncating='post', value=0)

#trainLSTM        = nn(train_X, train_y, correct_labels, dev_X, dev_y, correct_labels_dev, word_to_ix).trainingModel()  # 最优算法
trainLSTM_simple = nn_simple(train_X, train_y, correct_labels, dev_X, dev_y, correct_labels_dev, word_to_ix).trainingModel()  # 对比算法
path = r'.' # need to change the file path in windows
with open(os.path.join(path, '/trainHistoryDict'), 'wb') as file_pi:
        pickle.dump(trainLSTM.history, file_pi)

# generate precision recall f1score
def metrics_report(X,y):
    test_Y = np_utils.to_categorical(y, len(tag_to_ix)+1)
    modelpath = r'PDmodel_epoch_10_batchsize_32_embeddingDim_100_new2.h5'
    model = load_model(modelpath)
    score = model.evaluate(X, test_Y, batch_size=32)
    from sklearn import metrics
    pred = model.predict(X)
    pred_single = []
    test_Y_single = []
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_single.append(np.argmax(pred[i,j]))
            test_Y_single.append(np.argmax(test_Y[i,j]))
    print("acc\t%f\n"%(score[1]))
    print(metrics.classification_report(test_Y_single, pred_single))


#print("TRAIN METRICE ANALYSIS")
#metrics_report(train_X, train_y)
#print("DEV METRICE ANALYSIS")
#metrics_report(dev_X, dev_y)
#print("TEST METRICE ANALYSIS")
#metrics_report(test_X, test_y)


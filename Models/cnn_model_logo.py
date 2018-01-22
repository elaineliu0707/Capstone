#!/usr/bin python
from __future__ import print_function
import numpy as np
from gensim.models import Word2Vec
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import ast
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from functools import reduce
import os
from os.path import basename
import csv 
from keras import backend as K   
import glob
import random
 

random.seed(123456)
def generate_rand_text(words,max_len,count):
    if len(words)<max_len: 
        return [words]
    num_gen=(int)((len(words)/max_len)*count)
    #print(len(words),max_len,count,num_gen)
    rand_text=[]
    rand_len_max=len(words)-max_len
    for i in range(0,num_gen):
        rand_start=random.randint(0,rand_len_max)
        sub_text=words[rand_start:rand_start+max_len]
        #print(rand_start,rand_start+max_len)
        rand_text.append(sub_text)        
    return rand_text 
def generate_testing_text(words,max_len):
    if len(words)<max_len: 
        return [words]
    text_threshold=(int)(max_len/2)
    start_index=0
    end_index=max_len
    rand_text=[]
    text_len=len(words)
    while end_index<=text_len:
        #print(start_index,end_index)
        sub_text=words[start_index:end_index]
        start_index=start_index+text_threshold
        end_index=end_index+text_threshold
        rand_text.append(sub_text)
    # last patch
    if end_index!=(text_len+text_threshold):
        rand_text.append(words[text_len-max_len:text_len])
        #print(text_len-max_len,text_len)
    return rand_text 

embedding_dim = 8
filter_sizes = (3, 4)
num_filters = 250
#dropout_prob = (0.1, 0.75)
dropout_prob = (0.1, 0.35)
hidden_dims = 256 
sequence_length = 300 
# Training parameters
batch_size = 32
num_epochs = 20
#val_split = 0.33


done=[]
#done=["Alarabiya","Aljazeera","CNN","GA-on-Islamic-Affairs",
#      "Mohamed-Rateb-Al-Nabulsi","Movement-of-Society-for-Peace",
#      "Rabee-al-Madkhali","Salman-Fahd-Al-Ohda","Socialist-Union-Morocco",
#      "Tunisian-General-Union-of-Labor","Al-Boraq","Al-Shabaab",
#      "Ansar-Al-Sharia","AQIM","Azawad","Hamas","Hezbollah","Houthis",
#      "ISIS","Syrian-Democratic-Forces"]
 
TEXT_DATA_DIR='/sfs/lustre/scratch/ma2sm/arabic-docs_rand_train/*.txt'
Orig_TEXT_DATA_DIR='/sfs/lustre/scratch/ma2sm/arabic-docs/*.txt'

accuracy=[]
pre_0=[]
rec_0=[]
f1_0=[]
pre_1=[]
rec_1=[]
f1_1=[]
model_variation = 'CNN-rand'  #  CNN-rand | CNN-google | CNN-RPC
#print('Model variation is %s' % model_variation)

# Data Preparatopn
# ==================================================
#
# Load data
#print("Loading data...")
 

#load labels
filePath='/sfs/lustre/scratch/ma2sm/arabic-groups-labels.txt'
labels={}
with open(filePath,'r') as intputFile:
        reader=csv.reader(intputFile,delimiter=',')
        for fname,y in reader:
            labels[fname]=int(y)
# load all data
all_sentences = []  # list of text articles
#all_labels_index = {}  # dictionary mapping label name to numeric id
all_labels = []  # list of label ids
orig_all_sentences = []  # list of text articles
orig_all_labels = []  # list of label ids

#allFiles=sorted(os.listdir(TEXT_DATA_DIR))
all_docs_labels=[]
orig_all_docs_labels=[]

for fname in glob.iglob(TEXT_DATA_DIR):
#for fname in allFiles:
    #fpath = os.path.join(TEXT_DATA_DIR, fname)
    f = open(fname,'rb')
    all_sentences.append(f.read().decode('utf-8'))
    base_name=basename(fname)
    #all_labels_index[base_name] = len(all_labels_index)
    label=list(lbl for file,lbl in labels.items() if base_name.startswith(file))[0]
    all_labels.append(label)
    group=list(file for file,lbl in labels.items() if base_name.startswith(file))[0]
    all_docs_labels.append(group)
    f.close()
    
d = []
for i in all_sentences:
    words2 = text_to_word_sequence(i, lower=True, split=" ")
    d.append(words2)

all_sentences = d

# load orginal files
for fname in glob.iglob(Orig_TEXT_DATA_DIR):
#for fname in allFiles:
    #fpath = os.path.join(TEXT_DATA_DIR, fname)
    f = open(fname,'rb')
    orig_all_sentences.append(f.read().decode('utf-8'))
    base_name=basename(fname)
    #all_labels_index[base_name] = len(all_labels_index)
    label=list(lbl for file,lbl in labels.items() if base_name.startswith(file))[0]
    orig_all_labels.append(label)
    group=list(file for file,lbl in labels.items() if base_name.startswith(file))[0]
    orig_all_docs_labels.append(group)
    f.close()
    
d = []
for i in orig_all_sentences:
    words2 = text_to_word_sequence(i, lower=True, split=" ")
    d.append(words2)

orig_all_sentences = d
pos_groups_run=0
neg_groups_run=0
for grp_Name in labels:
    if grp_Name in done:
        continue
    if labels[grp_Name]==1:
        pos_groups_run=pos_groups_run+1
    else:
        neg_groups_run=neg_groups_run+1   
    train_sentences = []  # list of text articles
    train_labels = []  # list of label ids
    test_sentences = []  # list of text articles
    test_labels = []  # list of label ids
    vocab=set()
    for j in range(0,len(all_docs_labels)):
        if all_docs_labels[j]!=grp_Name:
            train_sentences.append(all_sentences[j])
            train_labels.append(all_labels[j])
    for j in range(0,len(orig_all_docs_labels)):
        if orig_all_docs_labels[j]==grp_Name:
            test_sentences.append(orig_all_sentences[j])
            test_labels.append(orig_all_labels[j])
        else:
            sub_vocab=set(orig_all_sentences[j])
            vocab=vocab|sub_vocab
            
    if len(test_labels)==0:
        continue
    print (grp_Name)
    vocab=list(vocab)
    #vocab2 = sorted(reduce(lambda x, y: x | y, (set(i) for i in train_sentences)))
    print("vocab len:%i"%len(vocab))
    # Reserve 0 for masking via pad_sequences
    # Reserve 1 for unseen testing tokens
    train_vocab_size = len(vocab) + 2
    train_word_idx = dict((c, i + 2) for i, c in enumerate(vocab))
    
    
    
    train_vocabulary= train_word_idx
    train_vocabulary_inv = vocab 
    train_vocabulary_inv.append("</PAD>")
    
    print ("Training: %i pos, %i neg, Testing: %i post, %i neg"%(train_labels.count(1), train_labels.count(0),
                                                                 test_labels.count(1), test_labels.count(0)))

    itr_accuracy=[]
    itr_pre_0=[]
    itr_rec_0=[]
    itr_f1_0=[]
    itr_pre_1=[]
    itr_rec_1=[]
    itr_f1_1=[]    
    
    oldStart=0
    
    for itr in range(0, 1):# to get better and more robust results, run each fold 10 times
      
        if model_variation=='CNN-google': 
            model_name='GoogleNews-vectors-negative300.bin'
            embedding_model = Word2Vec.load_word2vec_format(model_name, binary=True)
            embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in train_vocabulary_inv])]
        elif model_variation=='CNN-RPC':
            model_name='RPC.300'
            embedding_model = Word2Vec.load(model_name)
            embedding_weights = [np.array([embedding_model[w] if w in embedding_model\
                                                        else np.random.uniform(-0.25,0.25,embedding_model.vector_size)\
                                                        for w in train_vocabulary_inv])]
        elif model_variation=='CNN-rand':
            embedding_weights = None
        else:
            raise ValueError('Unknown model variation')    
             
        
      
    
        
        graph_in = Input(shape=(sequence_length, embedding_dim))
        convs = []
        for fsz in filter_sizes:
            conv = Convolution1D(nb_filter=num_filters,
                                 filter_length=fsz,
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1)(graph_in)
            pool = MaxPooling1D(pool_length=2)(conv)
            flatten = Flatten()(pool)
            convs.append(flatten)
        
        if len(filter_sizes)>1:
            out = Merge(mode='concat')(convs)
        else:
            out = convs[0]
        
        graph = Model(input=graph_in, output=out)
        
        # main sequential model
        model = Sequential()
        if not model_variation=='CNN-static':
            model.add(Embedding(len(train_vocabulary_inv)+1, embedding_dim, input_length=sequence_length,
                                weights=embedding_weights))
            model.add(Dropout(dropout_prob[0], input_shape=(sequence_length, embedding_dim)))
        model.add(graph)
        model.add(Dense(hidden_dims))
        model.add(Dropout(dropout_prob[1]))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        #
        
      # Training the model
        X = []
        for i in train_sentences:
            x = [train_word_idx[w] for w in i]
            X.append(x)
        
        X_train = pad_sequences(X,sequence_length)
        
        train_Y = np.asarray(train_labels) 
        model.fit(X_train, train_Y, batch_size=batch_size,nb_epoch=num_epochs, verbose=0 )#
        
        
 
        
        testing_probs=[]
        for i in range(len(test_sentences)):
            X = []
            testing_patches=generate_testing_text(test_sentences[i],sequence_length)
            for j in testing_patches:
                x = [train_word_idx[w] if w in train_word_idx else 1 for w in j]
                X.append(x)
        
            X_test = pad_sequences(X,sequence_length)
            test_count=len(X_test)
            probs = model.predict(X_test.reshape(test_count, -1))
            #print (probs)
            testing_probs.append(np.average(probs))
        test_count=len(test_sentences)    
        #print (testing_probs)
        testing_probs=np.asarray(testing_probs)
        test_Y = test_labels
        pred=(testing_probs>0.5).astype(int).flatten() 
        correct=np.sum(pred==test_Y)
        acc=correct/test_count
        
        CorrectClassifications=0
        PosClassified=0
        PosCorrectClassified=0
        NegClassified=0
        NegCorrectClassified=0 
        #Calculate Accuracy 
        j=0
        for  j in range(0,test_count) :
            true_label=test_Y[j]
            label=pred[j]    
            if(label==true_label):
                CorrectClassifications=CorrectClassifications+1
            if(label==1):
                PosClassified=PosClassified+1
            else:
                NegClassified=NegClassified+1
            if(label==1 and label==true_label):
                PosCorrectClassified=PosCorrectClassified+1 
            if(label==0 and label==true_label):
                NegCorrectClassified=NegCorrectClassified+1 
 
        # precision, recall & F-Measure
        TotalPos=test_Y.count(1)
        TotalNeg=test_Y.count(0)
      
        if PosClassified==0:
            posPre=0
        else:
            posPre=PosCorrectClassified/PosClassified
        if TotalPos==0:
            posRec=0
        else:   
            posRec=PosCorrectClassified/TotalPos
        if (posPre+posRec)==0:
            posF=0
        else: 
            posF= 2*posPre*posRec/(posPre+posRec) 
        if NegClassified==0:
            negPre=0
        else: 
            negPre= NegCorrectClassified/NegClassified 
        if TotalNeg==0:
            negRec=0
        else: 
            negRec =NegCorrectClassified/TotalNeg
        if (negPre+negRec)==0:
            negF=0
        else: 
            negF= 2*negPre*negRec/(negPre+negRec) 
        itr_accuracy.append(acc)
        itr_pre_0.append(negPre)
        itr_rec_0.append(negRec)
        itr_f1_0.append(negF)
        itr_pre_1.append(posPre)
        itr_rec_1.append(posRec)
        itr_f1_1.append(posF)
        K.clear_session()
        print("group %s itr %i, %i/%i, Accuracy:%f"%(grp_Name, itr, CorrectClassifications,test_count,acc))
        print("0 class (%i/%i/%i): precision:%f, recall:%f, fmeasure:%f"%(NegClassified,NegCorrectClassified,TotalNeg,negPre,negRec,negF))
        print("1 class (%i/%i/%i): precision:%f, recall:%f, fmeasure:%f"%(PosClassified,PosCorrectClassified,TotalPos,posPre,posRec,posF))
    
    log_file = open("cnn_log_logocv_aug.txt","a") 
    log_file.write("group %s\n"%grp_Name) 
    log_file.write("accuracy\n"+" ".join(['%f' % num for num in itr_accuracy])+"\n") 
    log_file.write("pre 0\n"+" ".join(['%f' % num for num in itr_pre_0])+"\n") 
    log_file.write("rec 0\n"+" ".join(['%f' % num for num in itr_rec_0])+"\n") 
    log_file.write("f1 0\n"+" ".join(['%f' % num for num in itr_f1_0])+"\n") 
    log_file.write("pre 1\n"+" ".join(['%f' % num for num in itr_pre_1])+"\n")  
    log_file.write("rec 1\n"+" ".join(['%f' % num for num in itr_rec_1])+"\n")  
    log_file.write("f1 1\n"+" ".join(['%f' % num for num in itr_f1_1])+"\n") 
    log_file.close()
    
    accuracy.append(np.average(itr_accuracy)) 
    pre_0.append(np.average(itr_pre_0))
    rec_0.append(np.average(itr_rec_0))
    f1_0.append(np.average(itr_f1_0))
    pre_1.append(np.average(itr_pre_1))
    rec_1.append(np.average(itr_rec_1))
    f1_1.append(np.average(itr_f1_1))
    
    print("group %s: , Accuracy:%f"%(grp_Name,np.average(itr_accuracy)))
    print("0 class: precision:%f, recall:%f, fmeasure:%f"%(np.average(itr_pre_0),np.average(itr_rec_0),np.average(itr_f1_0)))
    print("1 class: precision:%f, recall:%f, fmeasure:%f"%(np.average(itr_pre_1),np.average(itr_rec_1),np.average(itr_f1_1)))

print("Average:")  
print("Accuracy:%f"%(np.average(accuracy)))
print("0 class: precision:%f, recall:%f, fmeasure:%f"%(np.sum(pre_0)/neg_groups_run,np.sum(rec_0)/neg_groups_run,np.sum(f1_0)/neg_groups_run))
print("1 class: precision:%f, recall:%f, fmeasure:%f"%(np.sum(pre_1)/pos_groups_run,np.sum(rec_1)/pos_groups_run,np.sum(f1_1)/pos_groups_run))
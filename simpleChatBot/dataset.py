import json
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from utils import tokenizeWithLemma , bagOfWords
with open('intents.json' , 'r') as f:
    dic = json.load(f)

def preprocessing(dic):
    allWords , xy , tags = [] , [] , []
    for key in dic['intents'].keys():        
        tags.append( dic['intents'][key]['tag'] )   
        for pattern in dic['intents'][key]['patterns']:
            words = tokenizeWithLemma(pattern)
            xy.append((tags[-1] , words))
            allWords.extend(words)
            
    allWords = sorted(set(allWords))
    return allWords , xy , tags
    
def getXY(xy):   
    X_train , y_train = [] , []
    for ( tag , sentence ) in xy:
        bag = bagOfWords(sentence , dicOfWords)
        X_train.append(bag)
        y_train.append(tags.index(tag))   
    
    X_train = torch.tensor(np.array(X_train),dtype= torch.float32)
    y_train = torch.tensor(np.array(y_train) , dtype= torch.long)
    return X_train , y_train

def dicWords(allWords):
    dicOfWords = {}
    for i in range(len(allWords)):
        dicOfWords[allWords[i]] = i
    return dicOfWords

def getAll():
    return trainLoader  , len(dicOfWords)+1 , len(tags) , allWords , tags , dicOfWords

class ChatDataset(Dataset):
    
    def __init__(self , X , y):
        self.X = X
        self.y = y
        self.nSamples = len(y)
    
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self , idx):
        return self.X[idx] , self.y[idx]

allWords , xy , tags =preprocessing(dic)
dicOfWords = dicWords(allWords)
X_train , y_train = getXY(xy)
dataset = ChatDataset(X_train , y_train)
trainLoader = DataLoader(dataset = dataset , batch_size = 8 , shuffle = True )

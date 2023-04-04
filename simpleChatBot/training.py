import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from dataset import getAll
from model import NN

train_loader , input_size , num_classes , allWords ,tags , dicOfWords = getAll()
model = NN(input_size , 8 , num_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr = 1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer , mode = 'min' , factor = 0.7 , patience = 10 , verbose = False)

def train(model , train_loader , criterion , optimizer , device):
    trainLoss , acc = 0 , None
    correct , samples = 0 , 0
    for X , y in train_loader:
        X , y = X.to(device) , y.to(device)
        yPred = model(X)
        loss = criterion(yPred , y)
        trainLoss += loss.item()
        
        correct += (yPred.argmax(1) == y).sum()
        samples += y.shape[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    acc = np.round(correct/samples , 4)
    finalLoss = trainLoss / len(train_loader)
    return acc  , finalLoss

trainLoss = []
trainAcc =[]
for epoch in range(1111):
    acc , loss = train(model , train_loader , criterion , optimizer , device)
    trainLoss.append(loss)
    trainAcc.append(acc)
    scheduler.step(loss)
    if(epoch % 100 == 0):
        print( f'epoch : {epoch} | loss : {loss} | acc : {acc}' )
        
data = {
    'model_state' : model.state_dict() , 
    'input_size' : input_size ,
    'output_size' : num_classes ,
    'hidden_size' : 8 ,
    'all_words' : allWords ,
    'tags' : tags , 
    'dicOfWords' : dicOfWords
}

file = 'data.pth'
torch.save(data , file)
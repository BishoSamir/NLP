import numpy as np 
import json 
import torch 
from model import NN 
from utils import bagOfWords , tokenizeWithLemma
import re
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('intents.json' , 'r') as f:
    dic = json.load(f)
data = torch.load('data.pth')

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
dicOfWords = data['dicOfWords']
tags = data['tags']
model_state = data['model_state']

model = NN(input_size , hidden_size , output_size).to(device)
model.load_state_dict(model_state)
model.eval()
botName = 'Bot'
print("Let's chat! (type 'quit' to exit)")

while True :
    sentence = input('You: ')
    if sentence.lower() == 'quit':
        break
    
    sentence = tokenizeWithLemma(sentence)
    botInput = bagOfWords(sentence , dicOfWords)
    botInput = botInput.reshape(1 , -1)
    predicted = model( torch.from_numpy(botInput).float().to(device) )
    label = tags[ torch.argmax(predicted).item() ]
    
    for key in dic['intents'].keys():
        if dic['intents'][key]['tag'] == label:
            response = np.random.choice(dic['intents'][key]['responses'])
            print(f'{botName}: {response}')
            break
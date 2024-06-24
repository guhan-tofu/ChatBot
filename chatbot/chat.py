import random
import json
import torch
from nltk_utils import tokenize,bag_of_words
from model import NeuralNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' ) # using GPU

f = open('intents.json','r')
intents = json.load(f)

data = torch.load("data.pth",map_location='cpu')

model_state = data['model_state']
input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']

model = NeuralNetwork(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Hello There! type 'quit' to exit")

while True:
    sentence = input("You: ")
    if sentence.lower() == 'quit':
         print("Thank you for visiting Technocit")
         break
#def get_message(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0]) # 1 row and columns are now previous rows
    X = torch.from_numpy(X).to(device) # gives tensor from the numpy array
    # we are giving only one sample to the model each time
    output = model(X) # this contains the tensor of all the predictions for each class
    _,predicted = torch.max(output,dim=1) # getting maximum predicted class (dim=1 referes to columns)
    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()] # giees the probability of the label it is most likely to belong to

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent['tag']: # with our predicted label try to get the respones under that label
                #return random.choice(intent['responses']) # select any random response within the label
                print(random.choice(intent['responses'])," : ",prob.item())
    
    #return "Sorry, I do not understand..."
    else:
         print("Sorry, I do not understand...")

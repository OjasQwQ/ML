#Main Neural Network

import os
import random
import json

#Tokenisation & Lemmatisation 
import nltk

import numpy as np  

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
 
 #Architecture of the Neural Network
class ChatbotModel(nn.Module):                    #Inheriting from nn.Module

    def __init__ (self, input_size, output_size): #Class constructor
        super(ChatbotModel, self).__init__()      #Calling the constructor of the parent class nn.Module
        self.fc1 = nn.Linear(input_size, 128)     #First fully connected layer
        self.fc2 = nn.Linear(128, 64)             #Second fully connected layer
        self.fc3 = nn.Linear(64, output_size)     #Output layer
        self.relu=nn.ReLU()                       #Activation function
        self.dropout=nn.Dropout(p=0.5)            #50% dropout to prevent overfitting
    
    def forward(self, x):                        
        x = self.relu(self.fc1(x))               #First layer with ReLU activation
        x = self.dropout(x)                      #Dropout after first layer
        x = self.relu(self.fc2(x))              
        x= self.dropout(x)
        x = self.fc3(x)                          #Output layer
        return x                                 #Return output logits
        
 #Lemmatisation function
class ChatbotAssistant:

    def __init__(self, intents_path, function_mapping= None): 
        self.model= None
        self.intents_path= intents_path

        self.documents= []
        self.vocabulary= []
        self.intents=[]
        self.intents_responses=[]

        self.function_mapping= function_mapping

        self.X= None
        self.y= None

    @staticmethod
    def lemmatize_and_tokenize(text):
        lemmatizer= nltk.WordNetLemmatizer()       

        tokens= nltk.word_tokenize(text)           #Words
        tokens= [lemmatizer.lemmatize(word.lower()) for word in tokens] #Lemmatization & Lowercasing
        return tokens
    
    #BOW (Bag of Words) 
    @staticmethod
    def bag_of_words(tokens, vocabulary):
        return [1 if word in tokens else 0 for word in vocabulary]
    

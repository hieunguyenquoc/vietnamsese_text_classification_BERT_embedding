import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
from src.load_pretrained import load_pretrained_embedding
import numpy as np

class TextClassification(nn.Module):
    def __init__(self, args):
        super(TextClassification, self).__init__()
        self.pretrain_model_embedded = load_pretrained_embedding()
        self.args = args
        self.num_layers = args.num_layers
        self.input_embedd = args.num_words
        self.embedd_dim = self.pretrain_model_embedded.get_sentence_embedding_dimension()    #args.embedding_dim
        self.dropout = nn.Dropout(0.5)
        self.num_class = 18
        

        self.embedd = self.pretrain_model_embedded   #nn.Embedding(self.input_embedd, self.embedd_dim, padding_idx=0
        #self.embedd = nn.Embedding(self.input_embedd, self.embedd_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size = self.embedd_dim, hidden_size = self.embedd_dim, num_layers = self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(in_features=self.embedd_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_class)

    def forward(self,input):
        
        #print(type(input))
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        out = self.embedd.encode(input, convert_to_tensor=True)
        
        # out = torch.from_numpy(out)
        out = out.unsqueeze(1)
        #print(np.shape(out))
        # print(type(out))
        #print(np.shape(out))
        # print(out)
        # input = torch.tensor(input).to(device)
        #print(len(input))
        h = torch.zeros((self.num_layers,len(input),self.embedd_dim), device=device)
        c = torch.zeros((self.num_layers,len(input),self.embedd_dim), device=device)

        #out = self.embedd(input)
        
        out, (hidden, cell) = self.lstm(out, (h,c))

        out = self.dropout(out)
      
        out = torch.relu_(self.fc1(out[:,-1,:]))

        out = self.fc2(out)

        return out
    
        
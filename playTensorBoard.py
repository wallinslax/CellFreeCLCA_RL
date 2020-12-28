import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here

class actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(actor, self).__init__()

        self.state_dim = input_size
        self.action_dim = output_size
        self.h1_dim = 2*input_size###
        self.h2_dim = 2*input_size

        self.fc1 = nn.Linear(self.state_dim, self.h1_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(self.h1_dim)###
        self.ln1 = nn.LayerNorm(self.h1_dim)

        self.fc2 = nn.Linear(self.h1_dim, self.h2_dim)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(self.h2_dim)
        self.ln2 = nn.LayerNorm(self.h2_dim)

        self.fc3 = nn.Linear(self.h2_dim, self.action_dim)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, state):        
        #x = F.relu(self.bn1(self.fc1(state)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.ln1(x)
        #x = F.relu(x)
        x = F.tanh(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ln2(x)
        #x = F.relu(x)
        x = torch.tanh(x)

        x = self.fc3(x)
        #action = F.relu(x)
        action = torch.tanh(x)
        #action = F.softmax(x)
        #action = F.sigmoid(x)
        '''
        x = F.relu(self.ln1(self.bn1(self.fc1(state))))###
        x = F.relu(self.ln2(self.bn2(self.fc2(x))))
        action = F.relu(self.fc3(x))
        '''
        return action


net = actor(20,6)
alexnet = models.alexnet(pretrained=False)
alexnet.cuda()
#summary(alexnet, (3, 224, 224))
net.cuda()
print(net)
summary(net,(20),(6))
#print(alexnet)
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

writer.add_graph(net)
writer.close()
print('lala')
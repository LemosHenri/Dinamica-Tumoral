import numpy as np 
import matplotlib.pyplot as plt 
import torch 

from torch import nn
from classes import *

quantidade_camadas = 5
quantidade_neuronios = 300
amostras = [5_000, 10_000, 15_000, 20_000]

if __name__ == "__main__":

    for i in amostras:

        model = DeepONet(
            [100] + (quantidade_camadas + 1) * [quantidade_neuronios],
            [2] + (quantidade_camadas + 1) * [quantidade_neuronios],
            torch.relu
        )

        module = Module_DeepONet(model, nn.MSELoss())
        module.load_database('base dados/data20000_train.dat')
        module.trainning(i, 'modelos', 250, batch_size = 64, epochs = 1000, learning_rate = 1e-3, percent = 0.8)


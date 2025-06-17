import numpy as np 
import matplotlib.pyplot as plt 
import torch 

from torch import nn
from classes import *

quantidade_camadas = 5
quantidade_neuronios = 500
amostras = [100_000, 150_000, 200_000, 250_000, 300_000]

if __name__ == "__main__":

    for i in amostras:

        model = DeepONet(
            [84] + (quantidade_camadas + 1) * [quantidade_neuronios],
            [1] + (quantidade_camadas + 1) * [quantidade_neuronios],
            torch.relu
        )

        module = Module_DeepONet(model, nn.MSELoss())
        module.load_database('base dados/data300000.dat')
        module.trainning(i, 'modelos3', 100, batch_size = 128, epochs = 2000, learning_rate = 1e-2, percent = 0.7)


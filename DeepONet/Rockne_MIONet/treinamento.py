import torch 

from torch import nn
from classes import *

quantidade_camadas = 5
quantidade_neuronios = 200
amostras = [200_000, 250_000]

if __name__ == "__main__":

    for i in amostras:

        model = MIONet(
            [80] + (quantidade_camadas + 1) * [quantidade_neuronios],
            [4] + (quantidade_camadas + 1) * [quantidade_neuronios],
            [1] + (quantidade_camadas + 1) * [quantidade_neuronios],
            torch.relu
        )

        module = Module_MIONet(model, nn.MSELoss())
        module.load_database('base dados/data300000.dat')
        module.trainning(i, 'modelos', 250, batch_size = 64, epochs = 1000, learning_rate = 1e-3, percent = 0.8)


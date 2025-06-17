import numpy as np 
import matplotlib.pyplot as plt 
import torch 

from torch import nn
from classes import *

# Parâmetros
dosagem_max = [45, 65]               # Gy
periodo = [2, 8, 21, 35, 49]        # dias
fracionamento = [2, 6, 15, 25, 35]  # quantidade de fracionamentos
quantidade_protocolos = 10000        # por fracionamento
quantidade_nos = 500

if __name__ == "__main__":

    BaseDados = Database(fracionamento, periodo, dosagem_max = dosagem_max)

    BaseDados.generate_branch1(quantidade_protocolos, 80)
    BaseDados.generate_branch2([3.90e-03, 4.53e-02, 3.05e-02, 10.])
    BaseDados.generate_trunk(80)
    BaseDados.generate_target(np.exp(-100 * np.linspace(0, 1, quantidade_nos)**2), 20, 80)

    BaseDados.save(f'base dados/data{quantidade_protocolos * len(fracionamento)}.dat')
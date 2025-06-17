import numpy as np 
import matplotlib.pyplot as plt 
import torch 

from torch import nn
from classes import *

# Parâmetros
quantidade_condicoes_iniciais_treino = 20_000
quantidade_condicoes_iniciais_teste = quantidade_condicoes_iniciais_treino // 2

if __name__ == "__main__":

    #database = Database(quantidade_condicoes_iniciais_treino, 500, 80)
    #database.generate_data()
    #database.save(f'base dados/data{quantidade_condicoes_iniciais_treino}_train.dat')

    database = Database(quantidade_condicoes_iniciais_teste, 500, 80)
    database.generate_data()
    database.save(f'base dados/data{quantidade_condicoes_iniciais_teste}_test.dat')
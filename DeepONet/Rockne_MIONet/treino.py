import numpy as np
import matplotlib.pyplot as plt
import torch 

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from classes import *

def carregar_dados(arq: str):

    with open(arq, 'rb') as arq:

        branch1 = np.load(arq)
        branch2 = np.load(arq)
        trunk = np.load(arq)
        target = np.load(arq)

    return branch1, branch2, trunk, target

def divisao_dados(branch, target, percent = 0.8):

    branch = torch.from_numpy(branch)
    target = torch.from_numpy(target)

    div = int(len(branch) * percent)
    
    branch_train, branch_test, target_train, target_test = branch[:div], branch[div:], target[:div], target[div:]
    return MeuDataset(branch_train, target_train), MeuDataset(branch_test, target_test)

def train(dataloader, trunk, model, device, loss_fn, optimizer):

    model.train()
    loss_tl = 0

    for branch_batch, target_batch in dataloader:
            
        branch_batch1 = branch_batch[:, :-4].to(device)
        branch_batch2 = branch_batch[:, -4:].to(device)
        target_batch = target_batch.to(device)

        pred = model(branch_batch1, branch_batch2, trunk)
        loss = loss_fn(pred, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_tl += loss.item()

    return loss_tl / len(dataloader)

def test(dataloader, trunk, model, device, loss_fn, optimizer):

    model.eval()
    loss_tl = 0

    for branch_batch, target_batch in dataloader:
            
        branch_batch1 = branch_batch[:, :-4].to(device)
        branch_batch2 = branch_batch[:, -4:].to(device)
        target_batch = target_batch.to(device)

        pred = model(branch_batch1, branch_batch2, trunk)
        loss = loss_fn(pred, target_batch)

        loss_tl += loss.item()

    return loss_tl / len(dataloader)

# -------------------------------------------------------------
quantidade_neuronios = 400
quantidade_camadas_ocultas = 5
quantidade_amostras = 450_000
taxa_aprendizado = 1e-3
num_epochs = 2_000
batch = 2**6
arq_model = 'modelos2'
# --------------------- Sub-parâmetros ------------------------
device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
# -------------------------------------------------------------

branch1, branch2, trunk, target = carregar_dados('base dados/data450000x500.dat')
branch = np.hstack((branch1, branch2))

ds_train, ds_test = divisao_dados(branch[:quantidade_amostras], target[:quantidade_amostras])
dl_train, dl_test = DataLoader(ds_train, batch_size = batch, shuffle = True), DataLoader(ds_test, batch_size = batch, shuffle = True)
trunk = torch.from_numpy(trunk).to(device)

model = MIONet(
    [80] + (quantidade_camadas_ocultas + 1) * [quantidade_neuronios],    # Rede branch 1
    [4] + (quantidade_camadas_ocultas + 1) * [quantidade_neuronios],     # Rede branch 2
    [1] + (quantidade_camadas_ocultas + 1) * [quantidade_neuronios],     # Rede trunk
    torch.relu                  # Função de ativação
).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = taxa_aprendizado)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

loss_train = []
loss_test = []

for epoch in tqdm(range(1, num_epochs + 1), desc = f'Topologia: {quantidade_camadas_ocultas}x{quantidade_neuronios}  Amostras: {quantidade_amostras}'):

    ltrain = train(dl_train, trunk, model, device, loss_fn, optimizer)
    ltest = test(dl_test, trunk, model, device, loss_fn, optimizer)

    loss_train.append(ltrain)
    loss_test.append(ltest)

    scheduler.step()

    if epoch % 100 == 0: 

        local_m = f'{arq_model}/model{quantidade_neuronios}x{quantidade_camadas_ocultas}x{quantidade_amostras}x{epoch}.pt'
        local_t = f'{arq_model}/trainning{quantidade_neuronios}x{quantidade_camadas_ocultas}x{quantidade_amostras}x{epoch}.dat'


        loss_total = [loss_train, loss_test]

        model.save(local_m)
        with open(local_t, 'wb') as f: np.save(f, loss_total)
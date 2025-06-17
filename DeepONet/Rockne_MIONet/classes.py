import numpy as np 
import torch
from typing import Union, Optional
import random

from tqdm import tqdm 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

class Rockne_FDM:

    dx: float
    dt: float
    lb: float

    s: callable
    alpha: float
    alpha_beta: float

    def __init__(self, l: int, tf: int, d: float, p: float, alpha: float, alpha_beta: float, size_dom: int):

        self.l = l
        self.tf = tf
        self.d = d
        self.p = p
        self.size_dom = size_dom

        x = np.linspace(0, 1, size_dom, dtype=np.float32)
        t = np.linspace(0, tf * p, tf, dtype=np.float32)

        self.dx = x[1]
        self.dt = t[1]
        self.lb = (d * self.dt) / (p * l**2 * self.dx**2)

        self.s = lambda d, a, ab: np.exp(-a * (d + (d**2 / ab)))
        self.alpha = alpha 
        self.alpha_beta = alpha_beta

    def MLQ(self, d: float) -> float:

        if d == 0: return -1 
        return -1 + (1 - self.s(d, self.alpha, self.alpha_beta)) / self.p

    def __call__(self, x: np.ndarray, d: float = -1.0) -> np.ndarray:

        a = np.zeros((self.size_dom, self.size_dom))
        b = np.zeros((self.size_dom, self.size_dom))
        gamma_a = 1. + 0.5 * self.dt * self.MLQ(d) + self.lb
        gamma_b = 1. - 0.5 * self.dt * self.MLQ(d) - self.lb

        a[0,0], a[-1, -1] = -3 / (4 * self.dx), 3 / (4 * self.dx)
        a[0,1], a[-1, -2] =  1 / (self.dx), -1 / (self.dx)
        a[0,2], a[-1, -3] = -1 / (4 * self.dx), 1 / (4 * self.dx)

        b[0,0], b[-1, -1] =  3 / (4 * self.dx), -3 / (4 * self.dx)
        b[0,1], b[-1, -2] = -1 / (self.dx), 1 / (self.dx)
        b[0,2], b[-1, -3] =  1 / (4 * self.dx), -1 / (4 * self.dx)

        for i in range(1, self.size_dom - 1):
            a[i,i], b[i,i] = gamma_a, gamma_b
            a[i,i+1], b[i,i+1] = -0.5 * self.lb, 0.5 * self.lb
            a[i,i-1], b[i,i-1] = -0.5 * self.lb, 0.5 * self.lb

        dir = b @ x
        esq = np.linalg.solve(a, dir)
        return esq

class Database:

    branch1: np.array 
    branch2: np.array
    trunk: np.array
    target: np.array        

    def __init__(self, fracionamento: list[int], periodo: list[int], dosagem_max: Union[float, list[float]]):

        self.fracionamento = fracionamento 
        self.periodo = periodo 
        self.dosagem_max = dosagem_max

    def generate_branch1(self, data_size: int, array_size: int, shuffler: bool = True):

            protocolos = []

            # Validação dos inputs
            if len(self.fracionamento) != len(self.periodo):
                raise ValueError("fracionamento e periodo devem ter o mesmo tamanho")
            
            # Verifica se dosagem_max é um intervalo ou valor único
            if isinstance(self.dosagem_max, list):
                if len(self.dosagem_max) != 2:
                    raise ValueError("dosagem_max como lista deve ter exatamente 2 valores [min, max]")
                min_dose, max_dose = self.dosagem_max
            else:
                min_dose = max_dose = self.dosagem_max

            for num_frac, dias_trat in zip(self.fracionamento, self.periodo):
                if num_frac > dias_trat:
                    raise ValueError("O número de frações não pode ser maior que o número de dias de tratamento.")
                if dias_trat > array_size:
                    raise ValueError("O número total de dias de tratamento não pode exceder o tamanho do array.")
                
                for _ in range(data_size):
                    # Gera uma dosagem máxima aleatória para este protocolo
                    current_dosagem_max = random.uniform(min_dose, max_dose)
                    
                    # Inicializa o array com zeros
                    protocolo_array = [0] * array_size

                    # Gerar os dias de administração (5 dias úteis, 2 de descanso)
                    dias_fracoes = []
                    for semana_inicio in range(1, dias_trat + 1, 7):
                        dias_semana = list(range(semana_inicio, min(semana_inicio + 5, dias_trat + 1)))
                        dias_fracoes.extend(dias_semana)

                    # Selecionar dias aleatórios para as frações
                    dias_selecionados = sorted(random.sample(dias_fracoes, min(num_frac, len(dias_fracoes))))

                    # Gerar doses aleatórias
                    doses = [
                        round(random.uniform(0.1 * current_dosagem_max / num_frac, current_dosagem_max / num_frac), 2)
                        for _ in range(num_frac)
                    ]

                    # Ajustar para soma exata da dosagem máxima
                    fator_ajuste = current_dosagem_max / sum(doses)
                    doses_ajustadas = [round(dose * fator_ajuste, 2) for dose in doses]

                    # Preencher o protocolo
                    for dia, dose in zip(dias_selecionados, doses_ajustadas):
                        protocolo_array[dia - 1] = dose  # -1 para converter para índice zero-based

                    protocolos.append(protocolo_array)

            # Converter para numpy array e embaralhar se necessário
            protocolos_array = np.array(protocolos, dtype=np.float32)
            if shuffler:
                np.random.shuffle(protocolos_array)

            self.branch1 = protocolos_array

    def generate_branch2(self, params: Optional[list[float]] = None, alpha_range: list[float] = [0.025, 0.036], alpha_beta_range: list[float] = [1.0, 20.0], diffusion_range: list[float] = [0.00164, 0.8877], reaction_range: list[float] = [0.0027, 0.0887]):

        n_protocolos = len(self.branch1)
    
        if params is not None:
            # Usa os parâmetros fixos fornecidos
            branch2 = np.zeros((n_protocolos, len(params)), dtype=np.float32)
            for i in range(n_protocolos):
                branch2[i] = np.array(params, dtype=np.float32)
        else:
            # Gera parâmetros aleatórios dentro dos intervalos especificados
            alpha = np.random.uniform(alpha_range[0], alpha_range[1], size=n_protocolos)
            alpha_beta = np.random.uniform(alpha_beta_range[0], alpha_beta_range[1], size=n_protocolos)
            d = np.random.uniform(diffusion_range[0], diffusion_range[1], size=n_protocolos)
            p = np.random.uniform(reaction_range[0], reaction_range[1], size=n_protocolos)
            
            # Empilha os parâmetros em colunas
            branch2 = np.stack((d, p, alpha, alpha_beta), axis=1, dtype=np.float32)
        
        self.branch2 = branch2

    def generate_trunk(self, time: float):

        self.trunk = np.arange(0, time, 1.0).reshape(-1, 1)

    def calc_ratio(self, result: np.ndarray, lenght: float, percent: float):

            t_size, x_size = result.shape
            
            dominio = np.linspace(0, lenght, x_size)

            raio = np.zeros((t_size))
            for i in range(t_size): raio[i] = np.max(np.where(result[i] >= percent, dominio, 0))
            return raio
    
    def generate_target(self, initial_condition: np.array, lenght: float, time: int, percent: float = 0.6126):

        
        
        num_prot, num_it = self.branch1.shape 
        spatial_dom = len(initial_condition)

        target = np.zeros((num_prot, num_it), dtype = np.float32)

        for p in tqdm(range(num_prot)):

            rockne = Rockne_FDM(lenght, time, self.branch2[p, 0], self.branch2[p, 1], self.branch2[p, 2], self.branch2[p, 3], spatial_dom)
            result = np.zeros((num_it, spatial_dom))
            result[0] = initial_condition 

            for i in range(1, num_it): 
                
                if self.branch1[p,i-1] == 0.0: result[i] = rockne(result[i-1])
                else: result[i] = rockne(result[i-1], self.branch1[p,i-1])

            target[p] = self.calc_ratio(result, lenght, percent)

        self.target = target

    def save(self, local):

        with open(local, 'wb') as f:

            np.save(f, self.branch1)
            np.save(f, self.branch2)
            np.save(f, self.trunk)
            np.save(f, self.target)

class MIONet(nn.Module):

    activation_fn: callable
    loss_fn: callable
    device: str 

    def __init__(self, layers_branch1, layers_branch2, layers_trunk, activation_fn):

        super(MIONet, self).__init__()

        self.activation_fn = activation_fn
        self.loss_fn = nn.MSELoss()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

        self.branch1_net = nn.ModuleList()
        self.branch2_net = nn.ModuleList()
        self.trunk_net = nn.ModuleList()
        
        for i in range(len(layers_branch1)-1):
            self.branch1_net.append(nn.Linear(layers_branch1[i], layers_branch1[i+1]))
            nn.init.xavier_normal_(self.branch1_net[i].weight)

        for i in range(len(layers_branch2)-1):
            self.branch2_net.append(nn.Linear(layers_branch2[i], layers_branch2[i+1]))
            nn.init.xavier_normal_(self.branch2_net[i].weight)

        for i in range(len(layers_trunk)-1):
            self.trunk_net.append(nn.Linear(layers_trunk[i], layers_trunk[i+1]))
            nn.init.xavier_normal_(self.trunk_net[i].weight)

    
    def forward(self, input_branch1, input_branch2, input_trunk):

        u1, u2, u3 = input_branch1, input_branch2, input_trunk

        for layer in self.branch1_net: u1 = self.activation_fn(layer(u1))
        for layer in self.branch2_net: u2 = self.activation_fn(layer(u2))
        for layer in self.trunk_net: u3 = self.activation_fn(layer(u3))

        u = u1 * u2
        return torch.einsum('ai, bi -> ab', u, u3)
    
    def save(self, local: str):

        torch.save(self.state_dict(), local)

    def load(self, local: str):

        checkpoint = torch.load(local)
        self.load_state_dict(checkpoint)
        self.to(self.device)
        self.eval()

class MeuDataset(Dataset):

    def __init__(self, dados, rotulos):
        self.dados = dados
        self.rotulos = rotulos

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        amostra = self.dados[idx]
        rotulo = self.rotulos[idx]
        return amostra, rotulo
    
class Module_MIONet:

    def __init__(self, neural_network: MIONet, loss_function):

        self.loss_fn = loss_function
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

        self.neural_network = neural_network.to(self.device)
        
        self.loss_train = []
        self.loss_test = []

    def load_database(self, local):

        with open(local, 'rb') as arq:

            branch1 = np.load(arq)
            branch2 = np.load(arq)
            trunk = np.load(arq)
            target = np.load(arq)

        branch1 = np.array(branch1, dtype = np.float32)
        branch2 = np.array(branch2, dtype = np.float32)
        trunk = np.array(trunk, dtype = np.float32)
        target = np.array(target, dtype = np.float32)

        self.database = (branch1, branch2, trunk, target)
    
    def divisao_dados(self, size: int, batch_size: int = 64, percent = 0.8):

        branch = np.hstack((self.database[0][ :size], self.database[1][ :size]))

        branch = torch.from_numpy(branch)
        target = torch.from_numpy(self.database[-1][ :size])

        div = int(len(branch) * percent)
        
        branch_train, branch_test, target_train, target_test = branch[:div], branch[div:], target[:div], target[div:]
        dataset_train, dataset_teste = MeuDataset(branch_train, target_train), MeuDataset(branch_test, target_test)
        return DataLoader(dataset_train, batch_size = batch_size, shuffle = True), DataLoader(dataset_teste, batch_size = batch_size, shuffle = True)
    
    def train_instance(self, dataloader, trunk, optimizer):

        self.neural_network.train()
        loss_tl = 0

        for branch_batch, target_batch in dataloader:
                 
            branch_batch1 = branch_batch[:, :-4].to(self.device)
            branch_batch2 = branch_batch[:, -4:].to(self.device)
            target_batch = target_batch.to(self.device)

            pred = self.neural_network(branch_batch1, branch_batch2, trunk)
            loss = self.loss_fn(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tl += loss.item()

        return loss_tl / dataloader.dataset.__len__()
    
    def test_instane(self, dataloader, trunk, optimizer):

        self.neural_network.eval()
        loss_tl = 0

        for branch_batch, target_batch in dataloader:
                 
            branch_batch1 = branch_batch[:, :-4].to(self.device)
            branch_batch2 = branch_batch[:, -4:].to(self.device)
            target_batch = target_batch.to(self.device)

            pred = self.neural_network(branch_batch1, branch_batch2, trunk)
            loss = self.loss_fn(pred, target_batch)

            loss_tl += loss.item()

        return loss_tl / dataloader.dataset.__len__()
    
    def trainning(self, size: int, file_name:str, checkpoints: int, batch_size: int = 64, epochs: int = 1_000, learning_rate: float = 1e-3, percent: float = 0.8):
        
        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr = learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-4)
        batch_train, batch_test = self.divisao_dados(size, batch_size = batch_size, percent = percent)
        trunk = torch.from_numpy(self.database[2]).to(self.device)

        for ep in tqdm(range(1, epochs + 1), desc = 'Treinamento'):

            ltrain = self.train_instance(batch_train, trunk, optimizer)
            ltest = self.test_instane(batch_test, trunk, optimizer)

            self.loss_train.append(ltrain)
            self.loss_test.append(ltest)

            scheduler.step()

            if ep % checkpoints == 0:

                loss_total = [self.loss_train, self.loss_test]
                self.neural_network.save(file_name + f'/model{size}-{ep}.pt')
                with open(file_name + f'/trainning{size}-{ep}.dat', 'wb') as arq: np.save(arq, loss_total)
      
    def evaluate(self):

        self.neural_network.to('cpu')
        self.neural_network.eval()
        mean = lambda pred, ref: np.linalg.norm(pred - ref) / np.linalg.norm(ref)
        branch1, branch2, trunk = torch.from_numpy(self.database[0]), torch.from_numpy(self.database[1]), torch.from_numpy(self.database[2])
        pred = self.neural_network(branch1, branch2, trunk).detach().numpy()

        result = np.zeros(len(self.database[-1]))
        for i in range(len(self.database[-1])): result[i] = mean(pred[i], self.database[-1][i])
        return np.mean(result) * 100
    
    def get_results(self):

        self.neural_network.to('cpu')
        self.neural_network.eval()

        branch1, branch2, trunk, target = torch.from_numpy(self.database[0]), torch.from_numpy(self.database[1]), torch.from_numpy(self.database[2]), torch.from_numpy(self.database[-1])
        pred = self.neural_network(branch1, branch2, trunk).detach().numpy()
        return pred, target

    

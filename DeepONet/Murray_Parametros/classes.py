import numpy as np 
import torch
import gstools as gs

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

    branch: np.array 
    trunk: np.array
    target: np.array   

    def __init__(self, data_size: int, lenght_size: int, time_size: int):

        self.data_size = data_size
        self.lenght_size = lenght_size
        self.time_size = time_size

    def generate_initial_conditions(self, variancia: float = 1.0, correlacao: float = 0.1):

        initial_conditions = np.zeros((self.data_size, self.lenght_size))
        x = np.linspace(0, 1, self.lenght_size)
        for i in tqdm(range(self.data_size), desc = 'Gerando condições iniciais'):

            model = gs.Gaussian(dim=1, var=variancia, len_scale=correlacao)
            srf = gs.SRF(model)
            initial_conditions[i] = srf(x)

        return initial_conditions
    
    def generate_data(self, l = 20.0):

        difusion_coef = np.random.uniform(size = (self.data_size, 1), low = 0.0164, high = 0.8877)
        reaction_coef = np.random.uniform(size = (self.data_size, 1), low = 0.0027, high = 0.0877)
        self.branch = np.hstack((difusion_coef, reaction_coef), dtype = np.float32)

        x = np.linspace(0, 20, self.lenght_size)
        t = np.arange(0, self.time_size, 1.0)
        trunk = np.zeros((self.lenght_size * self.time_size, 2), dtype = np.float32)
        for i in range(self.time_size):
            for j in range(self.lenght_size):
                trunk[i * self.lenght_size + j] = [t[i], x[j]]

        self.trunk = trunk

        target = np.zeros((self.data_size, self.time_size * self.lenght_size), dtype = np.float32)
        inicial_condition = np.exp(-100 * ((x/l) - 0.5)**2)
        for i in tqdm(range(self.data_size), desc = 'Gerando dados de treinamento'):
            
            result = np.zeros((self.time_size, self.lenght_size))
            model = Rockne_FDM(l, self.time_size, self.branch[i, 0], self.branch[i, 1], 0.0305, 10., self.lenght_size)
            result[0] = inicial_condition
            for j in range(1, self.time_size): result[j] = model(result[j-1])
            target[i] = result.flatten()

        self.target = target

    def save(self, local):

        with open(local, 'wb') as f:

            np.save(f, self.branch)
            np.save(f, self.trunk)
            np.save(f, self.target)

class DeepONet(nn.Module):

    activation_fn: callable
    device: str 

    def __init__(self, layers_branch, layers_trunk, activation_fn):

        super(DeepONet, self).__init__()

        self.activation_fn = activation_fn
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

        self.branch_net = nn.ModuleList()
        self.trunk_net = nn.ModuleList()
        
        for i in range(len(layers_branch)-1):
            self.branch_net.append(nn.Linear(layers_branch[i], layers_branch[i+1]))
            nn.init.xavier_normal_(self.branch_net[i].weight)

        for i in range(len(layers_trunk)-1):
            self.trunk_net.append(nn.Linear(layers_trunk[i], layers_trunk[i+1]))
            nn.init.xavier_normal_(self.trunk_net[i].weight)
    
    def forward(self, input_branch, input_trunk):

        u1, u2 = input_branch, input_trunk

        for layer in self.branch_net: u1 = self.activation_fn(layer(u1))
        for layer in self.trunk_net: u2 = self.activation_fn(layer(u2))

        return torch.einsum('ai, bi -> ab', u1, u2)
    
    def save(self, local: str):

        torch.save(self.state_dict(), local)

    def load(self, local: str):

        checkpoint = torch.load(local)
        self.load_state_dict(checkpoint)
        self.to(self.device)
        self.eval()

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
    
class Module_DeepONet:

    def __init__(self, neural_network: DeepONet, loss_function):

        self.loss_fn = loss_function
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

        self.neural_network = neural_network.to(self.device)
        
        self.loss_train = []
        self.loss_test = []

    def load_database(self, local):

        with open(local, 'rb') as arq:

            branch = np.load(arq)
            trunk = np.load(arq)
            target = np.load(arq)

        branch = np.array(branch, dtype = np.float32)
        trunk = np.array(trunk, dtype = np.float32)
        target = np.array(target, dtype = np.float32)

        self.database = (branch, trunk, target)
    
    def divisao_dados(self, size: int, batch_size: int = 64, percent = 0.8):

        branch = self.database[0][ :size]
        target = self.database[-1][ :size]

        branch = torch.from_numpy(branch)
        target = torch.from_numpy(target)

        div = int(len(branch) * percent)
        
        branch_train, branch_test, target_train, target_test = branch[:div], branch[div:], target[:div], target[div:]
        dataset_train, dataset_teste = MeuDataset(branch_train, target_train), MeuDataset(branch_test, target_test)
        return DataLoader(dataset_train, batch_size = batch_size, shuffle = True), DataLoader(dataset_teste, batch_size = batch_size, shuffle = True)
    
    def train_instance(self, dataloader, trunk, optimizer):

        self.neural_network.train()
        loss_tl = 0

        for branch_batch, target_batch in dataloader:
                 
            branch_batch = branch_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            pred = self.neural_network(branch_batch, trunk)
            loss = self.loss_fn(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tl += loss.item()

        return loss_tl / dataloader.dataset.__len__()
    
    def test_instane(self, dataloader, trunk):

        self.neural_network.eval()
        loss_tl = 0

        for branch_batch, target_batch in dataloader:
                 
            branch_batch = branch_batch.to(self.device)
            target_batch = target_batch.to(self.device)

            pred = self.neural_network(branch_batch, trunk)
            loss = self.loss_fn(pred, target_batch)

            loss_tl += loss.item()

        return loss_tl / dataloader.dataset.__len__()
    
    def trainning(self, size: int, file_name:str, checkpoints: int, batch_size: int = 64, epochs: int = 1_000, learning_rate: float = 1e-3, percent: float = 0.7):
        
        optimizer = torch.optim.Adam(self.neural_network.parameters(), lr = learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-4)
        batch_train, batch_test = self.divisao_dados(size, batch_size = batch_size, percent = percent)
        trunk = torch.from_numpy(self.database[1]).to(self.device)

        for ep in tqdm(range(1, epochs + 1), desc = 'Treinamento'):

            ltrain = self.train_instance(batch_train, trunk, optimizer)
            ltest = self.test_instane(batch_test, trunk)

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
        branch, trunk = torch.from_numpy(self.database[0]), torch.from_numpy(self.database[1])
        pred = self.neural_network(branch, trunk).detach().numpy()

        result = np.zeros(len(self.database[-1]))
        for i in range(len(self.database[-1])): result[i] = mean(pred[i], self.database[-1][i])
        return np.mean(result) * 100
    
    def get_results(self):

        self.neural_network.to('cpu')
        self.neural_network.eval()

        branch, trunk, target = torch.from_numpy(self.database[0]), torch.from_numpy(self.database[1]), torch.from_numpy(self.database[-1])
        pred = self.neural_network(branch, trunk).detach().numpy()
        
        return pred, target

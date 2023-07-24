"""
Implementation of our patch-FNO for Darcy problem on L-shaped domain.
"""
import matplotlib.pyplot as plt
import numpy as np
import operator
from functools import reduce, partial
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# library for reading data
import mat73
# timer
from timeit import default_timer

#########################################
# nomi file da salvare
#########################################
#### Nomi file per tenosrboard e modello 
name_log_dir = 'exp_test'
name_model = 'model_test'

#########################################
# valori di default
#########################################
# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype
TrainDataPath = 'data/Darcy_Lshape_mp_solution_train.mat'
TestDataPath = 'data/Darcy_Lshape_mp_solution_test.mat'

#########################################
# variable and parameter to import data
#########################################
g = torch.Generator().manual_seed(1) # fisso seed
ntrain = 1000 # training instances
ntest = 200 # test instances
npatch = 2 # numero di patch

#########################################
# variable for training mesh
#########################################
multiple_mesh = False
if multiple_mesh:
    S = [2, 3, 5, 6]
    # Quando pongo multiple_mesh = True credo che n_train ed n_test devono essere
    # divisibili per batch_size*len(S), così che in ogni batch le dim di tutti i 
    # samples siano uguali, sennò dà errore.
else:
    s = 3 # parametro mesh: s=3 --> 70, s=1 --> 211

#########################################
# variable for loss function
#########################################
Loss = 'L2' # Scelgo quale funzione di loss minimizzare, posso usare o
# la norma H1 ('H1') relativa oppure la norma L2 relativa ('L2')
beta = 1 # calcola norma_{L_2} + beta*norma_{H_1}, tipo media pesata

#########################################
# activation function
#########################################
def activation(x):
    """
    Activation function che si vuole utilizzare all'interno della rete.
    La funzione è la stessa in tutta la rete.
    """
    return F.gelu(x)

# per kaiming initial normalization
fun_act = 'relu' 

#########################################
# hyperparameter for the neural operataor
#########################################   
# training hyperparameter   
# arc = 'Tran'
arc = 'Li'
learning_rate = 0.001 #l.r.
epochs = 5 # number of epochs for training
batch_size = 20
iterations = epochs*(ntrain//batch_size)
# model's hyperparameter
d_a = 3 # dimensione spazio di input
d_v = 32 # dimensione spazio nel Fourier operator
d_u = 1 # dimensione dell'output
L = 4 
padding_bool = False # False --> no padding
padding = 9
BN = False
# Kernel operator
weights_norm = 'Xavier'
# weights_norm = 'Kaiming'
modes = 12 # k_{max,j}
# FFTnorm = None 
FFTnorm = 'ortho'

#########################################
# tensorboard and plot variables
#########################################   
ep_step = 1 # ogni quante epoche salvare i plot su tensorboard
idx = [7, 42, 93, 158] # lista di numeri a caso tra 0 e n_test-1
# idx = [3, 7, 13, 19]
n_idx = len(idx)
plotting = True # Per stampare a video su terminale

#########################################
# nan for plt
#########################################
def set_nan(x):
    nx, ny = x.size()[-2], x.size()[-1]
    if len(x.size()) == 2:
        x[int((nx/2)):, int((ny/2)):] = np.nan 
    elif len(x.size()) == 3:
        x[:, int((nx/2)):, int((ny/2)):] = np.nan 
    elif len(x.size()) == 4:
        x[:, :, int((nx/2)):, int((ny/2)):] = np.nan 
    return x

#########################################
# somma di patch
#########################################
def sum_patch(x1, x2, continuation = False):
    """
    Funzione per sommare le soluzioni o input su due patch

    Parameters
    ----------
    x1 : tenosr
        valore sulla patch 1, dim = (n_batch)*(d_v)*(nx)*(ny)
    x2 : tensor
        valore sulla patch 2, dim = (n_batch)*(d_v)*(nx)*(ny)
    continuation: bool
        se True faccio il padding a zero, se è False restituisco il valore sui
        due domini separatemente con le stesse dim che hanno in input.
        Di default vale False

    Returns
    -------
    valore sul dominio totale

    """
    _, _, nx_1, ny_1 = x1.size()
    _, _, nx_2, ny_2 = x2.size()
    try:
        x1_sx = x1[:, :, :int(nx_1/2), :]
        x1_dx = x1[:, :, int(nx_1/2):, :]
        x2_tp = x2[:, :, :, :int(ny_2/2)]
        x2_bt = x2[:, :, :, int(ny_2/2):]
        x1 = torch.cat(((x1_sx + x2_tp)/2, x1_dx), dim = 2)
        x2 = torch.cat(((x1_sx + x2_tp)/2, x2_bt), dim = 3)
        if continuation:
            return torch.cat( (torch.cat(( (x1_sx + x2_tp)/2, x1_dx ), dim = 2), 
                               torch.cat(( x2_bt, torch.zeros_like(x2_bt) ), dim = 2)),
                               dim = 3)
        else:
            return x1, x2
    except:
        x1_tp = x1[:, :, :, :int(nx_1/2)]
        x1_bt = x1[:, :, :, int(nx_1/2):]
        x2_sx = x2[:, :, :int(ny_2/2), :]
        x2_dx = x2[:, :, int(ny_2/2):, :]
        x1 = torch.cat(((x2_sx + x1_tp)/2, x1_bt), dim = 3)
        x2 = torch.cat(((x2_sx + x1_tp)/2, x2_dx), dim = 2)
        if continuation:
            return torch.cat( (torch.cat(( (x2_sx + x1_tp)/2, x2_dx ), dim = 2), 
                               torch.cat(( x1_bt, torch.zeros_like(x1_bt) ), dim = 2)),
                               dim = 3)
        else:
            return x1, x2
        
#########################################
# reading data
#########################################
def MatReader(file_path):
    """
    Funzione per leggere i file di estensione .mat version 7.3

    Parameters
    ----------
    file_path : string
        path del file .mat da leggere        

    Returns
    -------
    a : tensor
        valutazioni della funzione a(x) del problema di Darcy 
        dimension = (n_samples)*(n_patch)*(nx)*(ny)
    u : tensor
        risultato dell'approssimazione della soluzione u(x) ottenuta con un
        metodo standard (nel nostro isogeometrico)
        dimension = (n_samples)*(n_patch)*(nx)*(ny)

    """
    data = mat73.loadmat(file_path)
    a = data["COEFF"]
    a = torch.from_numpy(a).float() # trasforma np.array in torch.tensor
    u = data["SOL"]
    u = torch.from_numpy(u).float()
    nodes = data["nodes"]
    nodes = torch.from_numpy(nodes).float()
    a, u, nodes = a.to('cpu'), u.to('cpu'), nodes.to('cpu')
    return a, u, nodes

#########################################
# loss function
#########################################
class L2relLoss():
    """ somma degli errori relativi in norma L2 """        
    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        
        return torch.sum(diff_norms/y_norms)
    
    def __call__(self, x, y):
        return self.rel(x, y)
    
class L2Loss():
    """ somma degli errori in norma L2 in dimensione d"""
    def __init__(self, d=2):
        self.d = d

    def rel(self, x, y):
        num_examples = x.size()[0]
        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        diff_norms = (h**(self.d/2))*torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)

        return torch.sum(diff_norms)

    def __call__(self, x, y):
        return self.rel(x, y)
    
class H1relLoss(object):
    """ Norma H^1 = W^{1,2} relativa, approssimata con la trasformata di Fourier """
    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        return torch.sum(diff_norms/y_norms)

    def __call__(self, x, y):
        nx = x.size()[1]
        ny = x.size()[2]
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),
                         torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),
                         torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])
        
        weight = 1 + beta*torch.sqrt(k_x**2 + k_y**2)
        loss = self.rel(x*weight, y*weight)

        return loss
    
#########################################
# initial normalization
#########################################    
class UnitGaussianNormalizer(object):
    """ 
    Initial normalization su x che è un tensore di 
    dimensione: (n_samples)*()*(nx)*(ny)
    normalization --> pointwise gaussian
    """
    def __init__(self, x, eps = 1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean)/(self.std + self.eps)
        return x

#########################################
# fourier layer
#########################################
class FourierLayer(nn.Module):
    """
    2D Fourier layer 
    input --> FFT --> linear transform --> IFFT --> output    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, FFTnorm):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply
        self.modes2 = modes2
        self.FFTnorm = FFTnorm
        
        # Meglio non usare i numeri complessi per i parametri
        # vedere https://github.com/pytorch/pytorch/issues/59998
        fourier_weight = [nn.Parameter(torch.FloatTensor(
            self.in_channels, self.out_channels, self.modes1, self.modes2, 2)) for _ in range(2)]
        self.fourier_weight = nn.ParameterList(fourier_weight)
        for param in self.fourier_weight:
            if weights_norm == 'Xavier':
                # Xavier normalization
                nn.init.xavier_normal_(param, gain = 1/(self.in_channels*self.out_channels))
            elif weights_norm == 'Kaiming':
                # Kaiming normalization
                torch.nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity=fun_act)
    
    @staticmethod
    def complex_matmul_2d(a, b):
        """ Moltiplicazione tra numeri complessi scomposti in parte reale ed immaginaria """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim = -1)

    def forward(self, x):
        B, I, M, N = x.shape
        
        # Trasformata di Fourier
        x_ft = torch.fft.rfft2(x, s = (M, N), norm = self.FFTnorm)
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim = 4) # a + ib
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        # Moltiplico i nodi significativi
        out_ft = torch.zeros(B, I, M, N // 2 + 1, 2, device=x.device)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.fourier_weight[0])
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_matmul_2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.fourier_weight[1])

        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1]) # Complex number
        
        # Inversa trasformata di Fourier
        x = torch.fft.irfft2(out_ft, s = (M, N), norm = self.FFTnorm)
        # x.shape == [batch_size, in_dim, grid_size, grid_size]
    
        return x
    
#########################################
# MLP
#########################################
class MLP(nn.Module):
    """ Rete neurale con un hidden layer (shallow neural network) """
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x) # affine transformation
        x = activation(x) # activation function
        x = self.mlp2(x) # affine transformation
        return x
    
#########################################
# Patch_FNO for Darcy on L-shaped
#########################################
class OverlappingFNO_DarcyL(nn.Module):
    """ 
    Fourier Neural Operator per il problema di Darcy in dimensione due    
    """
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2, BN, padding):
        """ 
        L: int
            numero di Fourier operator da fare
            
        d_a : int
            pari alla dimensione dello spazio in input
            
        d_v : int
            pari alla dimensione dello spazio nell'operatore di Fourier
            
        mode1 : int
            pari a k_{max, 1}
            
        mode2 : int
            pari a k_{max, 2}
            
        d_u : int
            pari alla dimensione dello spazio in output 
            
        BN: bool
            True se si vuole Batch Normalization sennò False
        """
        super(OverlappingFNO_DarcyL, self).__init__()
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.BN = BN
        self.padding_bool = padding_bool
        self.padding = padding
        self.FFTnorm = FFTnorm
        
        # Encoder
        self.p_1 = torch.nn.Conv2d(self.d_a, self.d_v, 1) # input features is d_a=3: (a(x, y), x, y)
        self.p_2 = torch.nn.Conv2d(self.d_a, self.d_v, 1)
        
        if arc == 'Tran': # pesi tipo alasdarian tran in ffno
            self.fouriers_1 = nn.ModuleList([]) # Fourier operator 
            self.fouriers_2 = nn.ModuleList([])
            self.ws1_1 = nn.ModuleList([])
            self.ws1_2 = nn.ModuleList([])
            self.ws2_1 = nn.ModuleList([])
            self.ws2_2 = nn.ModuleList([])
            self.BSs_1 = nn.ModuleList([])
            self.BSs_2 = nn.ModuleList([])
            for _ in range(self.L):
                # Fourier
                fourier_1 = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.fouriers_1.append(fourier_1)
                fourier_2 = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.fouriers_2.append(fourier_2)
                # Trasformazione lineare 1
                w1_1 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws1_1.append(w1_1)
                w1_2 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws1_2.append(w1_2)
                # Trasformazione lineare 2
                w2_1 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws2_1.append(w2_1)
                w2_2 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws2_2.append(w2_2)
                # Batch normalization
                if self.BN:
                    bs_1 = nn.BatchNorm2d(d_v)
                    self.BSs_1.append(bs_1)
                    bs_2 = nn.BatchNorm2d(d_v)
                    self.BSs_2.append(bs_2)
                    
        elif arc == 'Li' or arc == 'Residual': 
            # FNO originale proposta da zongy Li, oppure pesi tipo residual
            self.fouriers_1 = nn.ModuleList([]) # Fourier operator 
            self.fouriers_2 = nn.ModuleList([])
            self.ws_1 = nn.ModuleList([])
            self.ws_2 = nn.ModuleList([])
            self.BSs_1 = nn.ModuleList([])
            self.BSs_2 = nn.ModuleList([])
            for _ in range(self.L):
                # Fourier
                fourier_1 = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.fouriers_1.append(fourier_1)
                fourier_2 = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.fouriers_2.append(fourier_2)
                # Trasformazione lineare 1
                w_1 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws_1.append(w_1)
                w_2 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws_2.append(w_2)
                # Batch normalization
                if self.BN:
                    bs_1 = nn.BatchNorm2d(d_v)
                    self.BSs_1.append(bs_1)
                    bs_2 = nn.BatchNorm2d(d_v)
                    self.BSs_2.append(bs_2)
            
        # Decoder
        self.q_1 = MLP(self.d_v, self.d_u, 4*self.d_u) # output features is d_u: u(x, y)
        self.q_2 = MLP(self.d_v, self.d_u, 4*self.d_u)
        
    def forward(self, x_1, x_2):
        # x_1 e x_2 sono tensori di dim: (n_samples)*(n_patch)*(nx)*(ny)
        grid_1 = self.get_grid_1(x_1.shape, mydevice)
        x_1 = torch.cat((x_1, grid_1), dim = -1) # concateno lungo l'ultima dimensione
        
        grid_2 = self.get_grid_2(x_2.shape, mydevice)
        x_2 = torch.cat((x_2, grid_2), dim = -1) # concateno lungo l'ultima dimensione
        # ora x_1 e x_2 sono tensori di dim: (n_samples)*(nx)*(ny)*(3)
        
        x_1 = x_1.permute(0, 3, 1, 2) # (n_samples)*(3)*(nx)*(ny)
        x_2 = x_2.permute(0, 3, 1, 2) # (n_samples)*(3)*(nx)*(ny)
        
        # Applica P
        x_1 = self.p_1(x_1) # shape = (n_samples)*(d_v)*(nx)*(ny)
        x_2 = self.p_2(x_2) # shape = (n_samples)*(d_v)*(nx)*(ny)
        
        # Padding
        if self.padding_bool:
            x_1 = F.pad(x_1, [0, self.padding, 0, self.padding])
            x_2 = F.pad(x_2, [0, self.padding, 0, self.padding])
        
        # Fourier Layers
        for i in range(self.L):
            if arc == 'Tran':
                x_1, x_2 = sum_patch(x_1, x_2)
                # Prima componente
                xx_1 = self.fouriers_1[i](x_1)
                xx_1 = activation(self.ws1_1[i](xx_1))
                if self.BN: 
                    xx_1 = self.BSs_1[i](xx_1)
                xx_1 = self.ws1_2[i](xx_1)
                xx_1 = activation(xx_1) # 
                x_1 = x_1 + xx_1
                # Seconda componente
                xx_2 = self.fouriers_2[i](x_2)
                xx_2 = activation(self.ws2_1[i](xx_2))
                if self.BN: 
                    xx_2 = self.BSs_2[i](xx_2)
                xx_2 = self.ws2_2[i](xx_2)
                xx_2 = activation(xx_2) # 
                x_2 = x_2 + xx_2
            elif arc == 'Li':
                x_1, x_2 = sum_patch(x_1, x_2)
                # Prima componente
                x1_1 = self.fouriers_1[i](x_1)
                x2_1 = self.ws_1[i](x_1)
                x_1 = x1_1 + x2_1
                if self.BN: 
                    x_1 = self.BSs_1[i](x_1)
                if i < self.L - 1:
                    x_1 = activation(x_1)
                # Seconda componente
                x1_2 = self.fouriers_2[i](x_2)
                x2_2 = self.ws_2[i](x_2)
                x_2 = x1_2 + x2_2
                if self.BN: 
                    x_2 = self.BSs_2[i](x_2)
                if i < self.L - 1:
                    x_2 = activation(x_2)
            elif arc == 'Residual':
                x_1, x_2 = sum_patch(x_1, x_2)
                # Prima componente
                x_1_1 = self.fouriers_1[i](x_1)
                x_1_1 = self.ws_1[i](x_1_1)
                if self.BN: 
                    x_1_1 = self.BSs_1[i](x_1_1)
                if i < self.L - 1:
                    x_1 = x_1 + activation(x_1_1)
                # Seconda componente 
                x_2_2 = self.fouriers_2[i](x_2)
                x_2_2 = self.ws_2[i](x_2_2)
                if self.BN: 
                    x_2_2 = self.BSs_2[i](x_2_2)
                if i < self.L - 1:
                    x_2 = x_2 + activation(x_2_2)
                    
        # Padding
        if self.padding_bool:
            x_1 = x_1[..., :-self.padding, :-self.padding]
            x_2 = x_2[..., :-self.padding, :-self.padding]
        
        # applico Q
        x_1 = self.q_1(x_1) # shape = (n_samples)*(d_u)*(nx)*(ny)
        x_2 = self.q_2(x_2)
        
        x = sum_patch(x_1, x_2, True)
        
        x.permute(0, 2, 3, 1) # shape = (n_samples)*(nx)*(ny)*(d_u)
        
        return x.squeeze()
    
    def get_grid_1(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2] #n_sample, nx, ny
        # griglia x
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype = torch.float) # griglia uniforme
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1]) # adatto la dimensione
        # idem per la griglia y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype = torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(mydevice) #concateno lungo l'ultima dimensione
    
    def get_grid_2(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2] #n_sample, nx, ny
        # griglia x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype = torch.float) # griglia uniforme
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1]) # adatto la dimensione
        # idem per la griglia y
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype = torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(mydevice) #concateno lungo l'ultima dimensione


if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir )
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)
    
    #########################################
    # lettura dati e initial normalization
    ######################################### 
    
    #### Training data
    a_train, u_train, nodes = MatReader(TrainDataPath)
    idx_train = torch.randperm(a_train.size()[0], device = 'cpu', generator = g)[:ntrain]
    a_train, u_train = a_train[idx_train, :, :, :], u_train[idx_train, :, :, :]    
    a_train = torch.flip(a_train, [3])
    u_train = torch.flip(u_train, [3])
    # Gaussian pointwise normalization
    a_normalizer = UnitGaussianNormalizer(a_train) #compute mean e std
    a_train = a_normalizer.encode(a_train) # normalizzo
    
    #### Test data
    a_test, u_test, _ = MatReader(TestDataPath)
    idx_test = torch.randperm(a_test.size()[0], device = 'cpu', generator = g)[:ntest]
    a_test, u_test = a_test[idx_test, :, :, :], u_test[idx_test, :, :, :]
    a_test = torch.flip(a_test, [3])
    u_test = torch.flip(u_test, [3])
    # Gaussian pointwise normalization
    a_test = a_normalizer.encode(a_test) # normalize
    
    if multiple_mesh:
        l_train = ntrain//len(S)
        l_test = ntrain//len(S)
        for i, s in enumerate(S):
            try: # per quando ho n_patch > 1
                # extraction training data
                aa_train = a_train[l_train*i:l_train*(i+1), :, ::s, ::s]
                uu_train = u_train[l_train*i:l_train*(i+1), :, ::s, ::s]
                n_train, n_patch, nx_train, ny_train = aa_train.size()
                # extraction test data
                aa_test = a_test[l_test*i:l_test*(i+1), :, ::s, ::s]
                uu_test = u_test[l_test*i:l_test*(i+1), :, ::s, ::s]
                n_test,_ ,nx_test, ny_test = aa_test.size()
            except: # quando ho estratto una sola patch
                # extraction training data
                aa_train = a_train[l_train*i:l_train*(i+1), ::s, ::s]
                uu_train = u_train[l_train*i:l_train*(i+1), ::s, ::s]
                n_train, nx_train, ny_train = aa_train.size()
                
                # extraction test data
                aa_test = a_test[l_test*i:l_test*(i+1), ::s, ::s]
                uu_test = u_test[l_test*i:l_test*(i+1), ::s, ::s]
                n_test, nx_test, ny_test = aa_test.size()
            
            # aggiungo una dimensione, utile in seguito per quando unisco la griglia
            aa_train = aa_train.reshape(n_train, nx_train , ny_train, 1)
            aa_test = aa_test.reshape(n_test, nx_test, ny_test, 1)  
            
            if i == 0:
                # Creating dataset
                train_set = torch.utils.data.TensorDataset(aa_train, uu_train)
                test_set = torch.utils.data.TensorDataset(aa_test, uu_test)   
            else:
                # dataset
                train_set_2 = torch.utils.data.TensorDataset(aa_train, uu_train)
                test_set_2 = torch.utils.data.TensorDataset(aa_test, uu_test) 
                # Concatenate the dataset
                train_set = torch.utils.data.ConcatDataset([train_set, train_set_2])
                test_set = torch.utils.data.ConcatDataset([test_set, test_set_2])
        # Create the data loader dividing in minibatch
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size)    
             
    else:        
        # extraction training data
        a_train, u_train = a_train[:, :, ::s, ::s], u_train[:, :, ::s, ::s]        
        a_1_train = torch.cat((a_train[:, 1, :, :], a_train[:, 2, :, :]), dim = 1) # cat on x dim
        a_2_train = torch.cat((a_train[:, 1, :, :], a_train[:, 0, :, :]), dim = 2) #cat on y dim
        u_train = torch.cat( (torch.cat((u_train[:, 1, :, :], u_train[:, 2, :, :]), dim = 1), 
                              torch.cat((u_train[:, 0, :, :], torch.zeros_like(u_train[:, 0, :, :]))
                                        , dim = 1)), dim = 2)
        # extraction test data
        a_test, u_test = a_test[:, :, ::s, ::s], u_test[:, :, ::s, ::s]
        a_1_test = torch.cat((a_test[:, 1, :, :], a_test[:, 2, :, :]), dim = 1) # cat on x dim
        a_2_test = torch.cat((a_test[:, 1, :, :], a_test[:, 0, :, :]), dim = 2) #cat on y dim
        u_test = torch.cat( (torch.cat((u_test[:, 1, :, :], u_test[:, 2, :, :]), dim = 1), 
                              torch.cat((u_test[:, 0, :, :], torch.zeros_like(u_test[:, 0, :, :])),
                                        dim = 1)), dim = 2)
        
        # Add a final dimension, utile in seguito per quando unisco la griglia
        a_1_train = a_1_train.unsqueeze(-1)
        a_2_train = a_2_train.unsqueeze(-1)
        a_1_test = a_1_test.unsqueeze(-1)
        a_2_test = a_2_test.unsqueeze(-1)
        
        
        # Suddivisioni dei dati in batch
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(a_1_train, a_2_train, u_train),
            batch_size = batch_size)
        
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(a_1_test, a_2_test, u_test),
            batch_size = batch_size) 

    print('Data loaded')  
    
    ################################################################
    # training, evaluation e plot
    ################################################################
    
    # Inizializzazione del modello
    model = OverlappingFNO_DarcyL(d_a, d_v, d_u, L, modes, modes, BN, padding)
    # model.to(mydevice)
    
    # conta del numero di parametri utilizzati
    par_tot = 0
    for p in model.parameters():
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    print("Numero totale di parametri dell'operator network è:", par_tot)
    writer.add_text("Parametri", 'il numero totale di parametri è' + str(par_tot), 0)
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    # Funzione da minimizzare
    if Loss == 'L2':
        myloss = L2relLoss()
    elif Loss == 'H1':
        myloss = H1relLoss()
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        for a_1, a_2, u in train_loader:            
            a_1 = a_1.to(mydevice)
            a_2 = a_2.to(mydevice)
            u = u.to(mydevice)
            
            optimizer.zero_grad() # azzero il gradiente all'inizio per i vari batch
            out = model.forward(a_1, a_2) 
            if Loss == 'L2':
                loss = myloss(out.view(batch_size, -1), u.view(batch_size, -1))
            elif Loss == 'H1':
                loss = myloss(out, u)
            loss.backward()
    
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        test_h1 = 0.0
        with torch.no_grad(): # per efficienza
            for a_1, a_2, u in test_loader:
                a_1 = a_1.to(mydevice)
                a_2 = a_2.to(mydevice)
                u = u.to(mydevice)
    
                out = model.forward(a_1, a_2)      
                test_l2 += L2relLoss()(out.view(batch_size, -1), u.view(batch_size, -1)).item()
                test_h1 += H1relLoss()(out, u).item() 
                
        train_loss /= ntrain
        test_l2 /= ntest
        test_h1 /= ntest
    
        t2 = default_timer()
        print('Epoch:', ep,
              'Time:', t2-t1,
              'Train_loss:', train_loss,
              'Test_loss_l2:', test_l2, 
              'Test_loss_h1:', test_h1)
        writer.add_scalars('FNOplus_darcy2D', {'Train_loss': train_loss,
                                               'Test_loss_l2': test_l2,
                                               'Test_loss_h1': test_h1}, ep)
        
        #########################################
        # plot dei dati alla fine ogni ep_step epoche
        #########################################
        if ep == 0:
            #### Dato iniziale
            if multiple_mesh:
                s = min(S) # prendo la mesh più fine per fare l'esempio
                esempio_test = a_test[idx, ::s, ::s]
                n_esempio, nx_esempio, ny_esempio = esempio_test.size()
                # aggiungo una dimensione, utile in seguito per quando unisco la griglia
                esempio_test = esempio_test.reshape(n_esempio, nx_esempio, ny_esempio, 1) 
            else:
                esempio_1_test = a_1_test[idx]
                esempio_2_test = a_2_test[idx]
            
            esempio_prova = sum_patch(esempio_1_test.permute(0, 3, 1, 2),
                                      esempio_2_test.permute(0, 3, 1, 2), True)
            
            # Setto a white il colore del NaN.
            current_cmap = plt.cm.get_cmap("viridis").copy()
            current_cmap.set_bad(color = 'white')
            
            #### Coefficiente a(x)
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('coeff a(x)')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # x label
                ax[i].set(xlabel = 'x')
                # figura
                im = ax[i].imshow(set_nan(esempio_prova[i].squeeze()))
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('coeff a(x)', fig, 0)              
            
            #### Soluzione esatta
            if multiple_mesh:
                soluzione_test = u_test[idx, ::s, ::s]
            else:
                soluzione_test = u_test[idx]
            
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Soluzione esatta u(x)')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # x label
                ax[i].set(xlabel = 'x')
                # figura
                im = ax[i].imshow(set_nan(soluzione_test[i]))
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Soluzione esatta u(x)', fig, 0)
                
        # Soluzione approssimata dalla FNO e differenza
        if ep % ep_step == 0:
            with torch.no_grad(): # no grad per efficienza
                out_test = model(esempio_1_test, esempio_2_test)
                
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Soluzione approssimata con la FNO')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # x label
                ax[i].set(xlabel = 'x')
                # figura
                im = ax[i].imshow(set_nan(out_test[i]))
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Appro_sol', fig, ep)
            
            # Valore assoluto della differenza tra sol esatta ed approssimata
            diff = torch.abs(out_test - soluzione_test)
            
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Valore assoluto della differenza')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # x label
                ax[i].set(xlabel = 'x')
                # figura
                im = ax[i].imshow(set_nan(diff[i]))
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Diff', fig, ep)
    
    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    
    torch.save(model, name_model)   
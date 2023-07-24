"""
geoFNO per Airfoil problem, valutazione rete allenata
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
# timer
from timeit import default_timer

#########################################
# Parametri
#########################################
#### Nomi file per tenosrboard e modello 
Path = "model_test"
idx = [16, 45, 91, 155]
n_idx = len(idx)

#### Loss function
Loss = 'L2' # Scelgo quale funzione di loss minimizzare, posso usare o
# la norma H1 ('H1') relativa oppure la norma L2 relativa ('L2')
beta = 1 # calcola norma_{L_2} + beta*norma_{H_1}

#### Activation Function
def activation(x):
    """
    Activation function che si vuole utilizzare all'interno della rete.
    La funzione è la stessa in tutta la rete.
    """
    return F.gelu(x)
# fun_act = 'relu' # per kaiming initial normalization

#### Estrazione dei dati
Input_x = 'data/naca/NACA_Cylinder_X.npy'
Input_y = 'data/naca/NACA_Cylinder_Y.npy'
outout_sigma = 'data/naca/NACA_Cylinder_Q.npy'

g = torch.Generator().manual_seed(1) # fisso seed
ntrain = 1000 # training instances
ntest = 200 # test instances

t = 20 # tempi da discretizzare
s1 = 211 # numeri punti discretizzazoine profilo airfoil
s2 = 51 # numero di "strati" attorno alla airfoil
r1 = 1 # prendo un punto ogni r1 nei punti di s1
r2 = 1 # prendo uno "strato" ogni r2
# aggiorno i valori di s1 e s2 in base alle scelte di r1 e r2
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

#########################################
# iperparametri del modello
#########################################   
#### per il training     
learning_rate = 0.001 #l.r.
epochs = 5 # number of epochs for training
batch_size = 20
iterations = epochs*(ntrain//batch_size)
# per il modello
d_a = 4 # dimensione spazio di input
d_v = 32 # dimensione spazio nel Fourier operator
d_u = 1 # dimensione dell'output
L = 4 
modes1 = 12 # k_{max, 1}
modes2 = 12 # k_{max, 1}

#########################################
# valori di default
#########################################
# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

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
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)
        diff_norms = (h**(self.d/2))*torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)

        return torch.sum(diff_norms)

    def __call__(self, x, y):
        return self.rel(x, y)
    
class H1relLoss(object):
    """ Norma H^1 = W^{1,2} relativa, approssima con la trasformata di Fourier """
    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        return torch.sum(diff_norms/y_norms)

    def __call__(self, x, y):
        nx = x.size()[1]
        ny = x.size()[2]
        x = x.reshape(x.shape[0], nx, ny, -1)
        y = y.reshape(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
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
    dimensione: (n_samples)*(nx)*(ny)
    normalization --> pointwise gaussian
    """
    def __init__(self, x, eps = 1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

#########################################
# fourier layers
#########################################
class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer, self).__init__()
        """
        2D Fourier layer 
        input --> FFT --> linear transform --> IFFT --> output     
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

#########################################
# geoFNO for Euler's equation (airfoil)
#########################################
class Airfoil_geoFNO(nn.Module):
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2):
        super(Airfoil_geoFNO, self).__init__()
        """            
        d_a : int
            pari alla dimensione dello spazio in input
            
        d_v : int
            pari alla dimensione dello spazio nell'operatore di Fourier
            
        d_u : int
            pari alla dimensione dello spazio in output 
            
        L: int
            numero di Fourier operator da fare
            
        mode1 : int
            pari a k_{max, 1}
            
        mode2 : int
            pari a k_{max, 2}

        the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=211, y=51, c=4)
        output shape: (batchsize, x=211, y=51, t=20, c=4)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.padding = 8  # pad the domain if input is non-periodic
        
        self.P = nn.Conv2d(self.d_a, self.d_v, 1) # P
        
        # Fourier operator 
        self.fouriers = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        for _ in range(self.L):
            # Fourier
            fourier = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2)
            self.fouriers.append(fourier)
            # Trasformazione lineare 1
            w = nn.Conv2d(self.d_v, self.d_v, 1)
            self.ws.append(w)
        
        # Q
        self.Q1 = nn.Conv2d(self.d_v, 4*self.d_v, 1)
        self.Q2 = nn.Conv2d(4*self.d_v, self.d_u, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # x.shape = (batch, s1, s2, d_a)
        x = x.permute(0, 3, 1, 2) # x.shape = (batch, d_a, s1, s2)
        
        # P
        x = self.P(x) # x.shape = (batch, d_v, s1, s2)
        
        # pad the domain if input is non-periodic
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier Layers
        for i in range(self.L):
            x1 = self.fouriers[i](x)
            x2 = self.ws[i](x)
            x = x1 + x2
            if i < self.L - 1:
                x = activation(x)
        
        # pad the domain if input is non-periodic
        x = x[..., :-self.padding, :-self.padding]
        
        # Q
        x = self.Q1(x)
        x = activation(x)
        x = self.Q2(x) # x.shape = (batch, d_u, s1, s2)
        
        x = x.permute(0, 2, 3, 1) # x.shape = (batch, s1, s2, d_u)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # Griglia x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        # Griglia y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# 'cuda' se Ã¨ disponibile la GPU, sennÃ² Ã¨ 'cpu'
print('Device disponibile:', mydevice)
################################################################
# load data
################################################################
inputX = np.load(Input_x)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(Input_y)
inputY = torch.tensor(inputY, dtype=torch.float)
Input = torch.stack([inputX, inputY], dim=-1)

output = np.load(outout_sigma)[:, 4]
output = torch.tensor(output, dtype=torch.float)

a_train = Input[:ntrain, ::r1, ::r2][:, :s1, :s2]
u_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]

a_test = Input[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
u_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]

print("Data loaded")

#########################################
# importo il modello
#########################################
# per importare il modello serve definire la classe FNOplus_Darcy2d
model = torch.load(Path)
model.eval()

# conta del numero di parametri utilizzati
par_tot = 0
for p in model.parameters():
    # print(p.shape)
    par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
print("Numero totale di parametri dell'operator network Ã¨:", par_tot)
    
    
#########################################
# plot dei dati alla fine ogni ep_step epoche
#########################################

# Estraggo solo pochi punti attorno al profilo della airfoil
# per una migliore visualizzazione
nx = 40//r1
ny = 20//r2
size = 8

#### Dato iniziale (mesh)
in_test = a_test[idx]
#### Valori della mesh
X = a_test[idx, : ,:, 0].squeeze()
Y = a_test[idx, :, :, 1].squeeze()
# Estraggo solo i più vicini alla airfoil
X_small = X[:, nx:-nx, :ny]
Y_small = Y[:, nx:-nx, :ny]

fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
fig.suptitle('Mesh iniziale')
ax[0].set(ylabel = 'y')
for i in range(n_idx):
    ax[i].set(xlabel = 'x')
    # figura del dato iniziale
    ax[i].scatter(X_small[i], Y_small[i], size)            
plt.show()
    
#### Soluzione esatta
sol_test = u_test[idx]
fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
fig.suptitle('Soluzione esatta')
ax[0].set(ylabel = 'y')
for i in range(n_idx):
    ax[i].set(xlabel = 'x')
    # Turn off tick labels
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    # figura
    im = ax[i].pcolormesh(X_small[i], Y_small[i], sol_test[i][nx:-nx, :ny], shading='gouraud') 
    fig.colorbar(im, ax = ax[i])
plt.show()
    
#### Soluzione approssimata
t1 = default_timer()
with torch.no_grad():  # efficiency
    out_test = model(in_test).squeeze()
t2 = default_timer()
print('Tempo impiegato nel calcolo delle soluzioni approssimate con geoFNO:', t2-t1)

fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
fig.suptitle('Soluzione approssimata')
ax[0].set(ylabel = 'y')
for i in range(n_idx):
    ax[i].set(xlabel = 'x')
    # Turn off tick labels
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    # figura
    im = ax[i].pcolormesh(X_small[i], Y_small[i], out_test[i][nx:-nx, :ny], shading='gouraud')     
    fig.colorbar(im, ax = ax[i])
plt.show()

#### Differenza
diff = np.abs(sol_test - out_test)
fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
fig.suptitle('Differenza')
ax[0].set(ylabel = 'y')
for i in range(n_idx):
    ax[i].set(xlabel = 'x')
    # Turn off tick labels
    ax[i].set_yticklabels([])
    ax[i].set_xticklabels([])
    # figura
    im = ax[i].pcolormesh(X_small[i], Y_small[i], diff[i][nx:-nx, :ny], shading='gouraud') 
    fig.colorbar(im, ax = ax[i])
plt.show()


for i in range(n_idx):
    print('Esempio:', i)
    # errore in norma L2
    err_L2 = L2Loss()(sol_test[[i]], out_test[[i]]).item()
    print('Errore in norma L2 =', err_L2)
    # errore relativo in norma L2
    err_rel_L2 = L2relLoss()(sol_test[[i]], out_test[[i]]).item()
    print('Errore relativo in norma L2 =', err_rel_L2)
    # errore relativo in norma H1
    # err_rel_H1 = H1relLoss()(sol_test, out_test.unsqueeze(0)).item()
    # print('Errore relativo in norma H1 =', err_rel_H1)
    # errore relativo in norma l_{inf}
    err_l_inf = torch.max(diff[i]).item()
    print('Errore relativo in norma l_inf =', err_l_inf)

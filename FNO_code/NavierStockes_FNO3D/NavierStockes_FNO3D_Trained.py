"""
Test FNO precedentemente allenate
"""
import matplotlib.pyplot as plt
import numpy as np
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
# library for reading data
import scipy.io
import h5py
# timer
from timeit import default_timer

#########################################
# valori di default e modello
#########################################
# scelgo il modello da importare
Path = "model_test"
# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

#########################################
# importo train e test dataset
#########################################
def MatReader(file_path):
    """
    Funzione per leggere i file di estensione .mat per il problema di Navier-Stockes

    Parameters
    ----------
    file_path : string
        path del file .mat da leggere        

    Returns
    -------
    a : tensor
        valutazioni della funzione soluzione del problema di Navier Stockes 
        per tutti i valori dei tempi

    """
    try:
        data = scipy.io.loadmat(file_path)
    except:
        data = h5py.File(file_path)
    a = data["u"]
    a = a[()]
    a = np.transpose(a, axes=range(len(a.shape) -1, -1, -1))
    a = torch.from_numpy(a).float() # trasforma np.array in torch.tensor
    a = a.to('cpu')
    return a

#########################################
# activation function
#########################################
def activation(x):
    """
    Activation function che si vuole utilizzare all'interno della rete.
    La funzione è la stessa in tutta la rete.
    """
    return F.relu(x)

# per kaiming initial normalization
fun_act = 'relu' 

#########################################
# loss functions
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
# MLP
#########################################
class MLP(nn.Module):
    """ Rete neurale con un hidden layer (shallow neural network) """
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x) # affine transformation
        x = activation(x) # activation function
        x = self.mlp2(x) # affine transformation
        return x

#########################################
# fourier layer
#########################################
class FourierLayer(nn.Module):
    """
    3D Fourier layer 
    input --> FFT --> linear transform --> IFFT --> output    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply
        self.modes2 = modes2
        self.modes3 = modes3
            
        # Weights with kaiming initilization
        self.weights1 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)),
            a=0, mode = 'fan_in', nonlinearity = fun_act)
        self.weights2 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)),
            a = 0, mode = 'fan_in', nonlinearity = fun_act)  
        self.weights3 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)),
            a = 0, mode = 'fan_in', nonlinearity = fun_act) 
        self.weights4 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)),
            a = 0, mode = 'fan_in', nonlinearity = fun_act) 
        
    # Complex multiplication
    def compl_mul3d(self, input, weights):
        """ Moltiplicazione tra numeri complessi """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Trasformata di Fourier 2D per segnali reali
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        #### Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Trasformata inversa di Fourier 2D per segnali reali
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

#########################################
# FNO 3D
#########################################
class FNO_NavierStockes_3d(nn.Module):
    """ 
    Fourier Neural Operator per problema in 3d.
    input della rete è:
        (n_samples)*(n_x)*(n_y)*(T)*(T_in + 3)
    output della rete è:
        (n_samples)*(n_x)*(n_y)*(T)*(1)
    """
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2, modes3, BN):
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
            
        mode3 : int
            pari a k_{max, 3}
            
        d_u : int
            pari alla dimensione dello spazio in output 
            
        BN: bool
            True se si vuole Bathc Normalization sennò False
        """
        super(FNO_NavierStockes_3d, self).__init__()
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.BN = BN
        
        # Encoder
        self.p = nn.Linear(self.d_a, self.d_v) # input features is d_a
        
        # Fourier operator
        self.fouriers = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.BSs = nn.ModuleList([])
        for _ in range(self.L):
            # Fourier
            fourier = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.modes3)
            self.fouriers.append(fourier)
            # Trasformazione lineare
            w = nn.Linear(self.d_v, self.d_v)
            self.ws.append(w)
            # Batch normalization
            if self.BN:
                bs = nn.BatchNorm2d(d_v)
                self.BSs.append(bs)
                
        # Decoder
        self.q = nn.Linear(self.d_v, self.d_u) # output features is d_u: u(x, y)
        

    def forward(self, x):
        grid = self.get_grid(x.shape, mydevice)
        x = torch.cat((x, grid), dim = -1) #concateno lungo l'ultima dimensione
        # ora x è un tensore di dimensione: (n_samples)*(n_x)*(n_y)*(T)*(13)
        
        # Applica P
        x = self.p(x)
        
        # Riordina le features (n_samples)*(d_v)*(n_x)*(n_y)*(T)
        x = x.permute(0, 4, 1, 2, 3)
        
        # Fourier Layers
        for i in range(self.L):
            x1 = self.fouriers[i](x)
            x2 = self.ws[i](x.permute(0, 2, 3, 4, 1))
            x = x1 + x2.permute(0, 4, 1, 2, 3)
            if self.BN: 
                x = self.BSs[i](x)
            if i < self.L - 1:
                x = activation(x)
        
        # applico Q
        x = self.q(x.permute(0, 2, 3, 4, 1))
        return x.squeeze() # tolgo l'ultima dimensione
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_t = shape[0], shape[1], shape[2], shape[3]
        # griglia x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_t, 1])
        # griglia y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_t, 1])
        # griglia t
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, size_t, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridt), dim=-1).to(device)

#########################################
# lettura dati e initial normalization
######################################### 
s = 1 # parametro mesh: s=1 --> 64
T_in = 10 # come input prendo il problema per 0:T_in-1
T = 40 # tempi che voglio predire
step = 1 # periodo di cui faccio la previsione

ntrain = 1000 # training instances
ntest = 200 # test instances

# Indici
g = torch.Generator().manual_seed(1) # fisso seed
indexes = torch.randperm(5000, device = 'cpu', generator = g)
idx_train = indexes[:ntrain]
idx_test = indexes[ntrain:ntrain + ntest]

# Training data
TrainDataPath = 'data/ns_V1e-3_N5000_T50.mat'
a = MatReader(TrainDataPath)
a_train, u_train = a[idx_train, :, :, :T_in], a[idx_train, :, :, T_in:T_in + T]    
# Gaussian pointwise normalization
a_normalizer = UnitGaussianNormalizer(a_train) # compute mean e std
a_train = a_normalizer.encode(a_train) # normalizzo
# extraction data
a_train, u_train = a_train[:, ::s, ::s, :], u_train[:, ::s, ::s, :]
_, nx_train, ny_train, _ = a_train.size()

# Test data
a_test, u_test = a[idx_test, :, :, :T_in], a[idx_test, :, :, T_in:T_in + T]  
# Gaussian pointwise normalization
a_test = a_normalizer.encode(a_test) # normalize
# extraction data
a_test, u_test = a_test[:, ::s, ::s, :], u_test[:, ::s, ::s, :]
_, nx_test, ny_test, _ = a_test.size()

# Aggiungo dimensione
a_train = a_train.reshape(ntrain,nx_train,ny_train,1,T_in).repeat([1,1,1,T,1])
a_test = a_test.reshape(ntest,nx_test,ny_test,1,T_in).repeat([1,1,1,T,1])

print('Data loaded')

#########################################
# importo il modello
#########################################
# per importare il modello serve definire la classe FNO_Darcy2d
model = torch.load(Path)
model.eval() 

#########################################
# esempio di zero-shot super resolution
#########################################
# stampo a video dato iniziale
idx = [42]
esempio_test = a_test[idx]
esempio_test_squeeze = esempio_test[:, :, :, 0, :].squeeze(3)
im = plt.imshow(esempio_test_squeeze[..., 0].squeeze()) # prima osservazione
plt.colorbar()
plt.show() 

# stampo a video sol esatta
soluzione_test = u_test[idx]
im = plt.imshow(soluzione_test[..., -1].squeeze()) # ultima osservazione
plt.colorbar(im)
plt.show()

# calcolo soluzione approssimata con la FNO
t1 = default_timer()
with torch.no_grad(): # no grad per efficienza
    out_test = model(esempio_test)
t2 = default_timer()
print('Tempo impiegato nel calcolo della soluzione approssimata con FNO:', t2-t1)
    
# stampo a video output FNO
im = plt.imshow(out_test[..., -1].squeeze()) # ultima osservazione
plt.colorbar(im)
plt.show()
   
# Valore assoluto della differenza tra sol esatta ed approssimata
diff = torch.abs(out_test - soluzione_test)
# stampo a video differenza tra sol esatta ed approssimata
im = plt.imshow(diff[..., -1].squeeze()) # ultima differenza
plt.colorbar(im)
plt.show()

# errore in normal L2
loss = L2Loss() #TODO fix it(?)
err_L2 = loss(soluzione_test, out_test).item()
print('Errore in norma L2 =', err_L2)
# errore relativo in norma L2
loss_rel = L2relLoss()
err_rel_L2 = loss_rel(soluzione_test, out_test).item()
print('Errore relativo in norma L2 =', err_rel_L2) 


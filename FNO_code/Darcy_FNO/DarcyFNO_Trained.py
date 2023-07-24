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
# timer
from timeit import default_timer

#########################################
# valori di default e modello
#########################################
# scelgo il modello da importare
Path = "model_L4"
# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

#########################################
# importo train e test dataset
#########################################
def MatReader(file_path):
    """
    Funzione per leggere i file di estensione .mat per il problema di Darcy

    Parameters
    ----------
    file_path : string
        path del file .mat da leggere        

    Returns
    -------
    a : tensor
        valutazioni della funzione a(x) del problema di Darcy
    u : tensor
        risultato dell'approssimazione della soluzione u(x) ottenuta con un
        metodo standard (nel nostro caso differenze finite con 421 punti)

    """
    data = scipy.io.loadmat(file_path)
    a = data["coeff"]
    a = torch.from_numpy(a).float() # trasforma np.array in torch.tensor
    u = data["sol"]
    u = torch.from_numpy(u).float()
    a, u = a.to('cpu'), u.to('cpu')
    return a, u

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
    2D Fourier layer 
    input --> FFT --> linear transform --> IFFT --> output    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply
        self.modes2 = modes2
            
        # Weights with kaiming initilization
        self.weights1 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
            a=0, mode = 'fan_in', nonlinearity = fun_act)
        self.weights2 = torch.nn.init.kaiming_normal_(
            nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
            a = 0, mode = 'fan_in', nonlinearity = fun_act)        

    def compl_mul2d(self, input, weights):
        """ Moltiplicazione tra numeri complessi """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Trasformata di Fourier 2D per segnali reali
        x_ft = torch.fft.rfft2(x)

        #### Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # angolo in alto a sx
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # angolo in basso a sx
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Trasformata inversa di Fourier 2D per segnali reali
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

#########################################
# FNO per Darcy
#########################################
class FNO_Darcy2d(nn.Module):
    """ 
    Fourier Neural Operator per il problema di Darcy in dimensione due    
    """
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2, BN):
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
            True se si vuole Bathc Normalization sennò False
        """
        super(FNO_Darcy2d, self).__init__()
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.BN = BN
        
        # Encoder
        self.p = nn.Linear(self.d_a, self.d_v) # input features is d_a=3: (a(x, y), x, y)
        
        # Fourier operator
        self.fouriers = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.BSs = nn.ModuleList([])
        for _ in range(self.L):
            # Fourier
            fourier = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2)
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
        x.unsqueeze(3)
        x = torch.cat((x, grid), dim = -1) #concateno lungo l'ultima dimensione
        # ora x è un tensore di dimensione: (n_samples)*(n_x)*(n_y)*(3)
        
        # Applica P
        x = self.p(x)
        
        # Riordina le features (n_samples)*(d_v)*(n_x)*(n_y)
        x = x.permute(0, 3, 1, 2)
        
        # Fourier Layers
        for i in range(self.L):
            x1 = self.fouriers[i](x)
            x2 = self.ws[i](x.permute(0, 2, 3, 1))
            x = x1 + x2.permute(0, 3, 1, 2)
            if self.BN: 
                x = self.BSs[i](x)
            if i < self.L - 1:
                x = activation(x)
        
        # applico Q
        x = self.q(x.permute(0, 2, 3, 1))
        return x # x.squeeze() se volgio togliere l'uno finale
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2] #n_sample, n_x, n_y
        # griglia x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype = torch.float) # griglia uniforme
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1]) # adatto la dimensione
        # idem per la griglia y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype = torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(mydevice) #concateno lungo l'ultima dimensione

#########################################
# lettura dati e initial normalization
#########################################
# train dataset
s = 1 # parametro mesh
idx = 42 # numero a caso per fare il test

# Training data
ntrain = 1000 # training instances
g = torch.Generator().manual_seed(1) # stesso seed della fase di train
idx_train = torch.randperm(1024, device = 'cpu', generator = g)[:ntrain]
TrainDataPath = 'data/piececonst_r421_N1024_smooth1.mat'
a_train, u_train = MatReader(TrainDataPath)
a_train, u_train = a_train[idx_train, ::, ::], u_train[idx_train, ::, ::]    
# Gaussian pointwise normalization
a_normalizer = UnitGaussianNormalizer(a_train) #compute mean e std

# test dataset
TestDataPath = 'data/piececonst_r421_N1024_smooth2.mat'
a_test, u_test = MatReader(TestDataPath)
a_test = a_normalizer.encode(a_test) #normalization
# extraction data
a_test, u_test = a_test[idx, ::s, ::s], u_test[idx, ::s, ::s]
nx_test, ny_test = a_test.size()

print("Data loaded")

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
im = plt.imshow(a_test)
plt.colorbar()
plt.show() 

# stampo a video sol esatta
im = plt.imshow(u_test)
plt.colorbar(im)
plt.show()

t1 = default_timer()
with torch.no_grad(): # no grad per efficienza
    # sistemo dim (1 samples, e 1 in fondo per la griglia)
    out_test = model.forward(a_test.unsqueeze(0).unsqueeze(3))
    out_test = out_test.squeeze()
    
t2 = default_timer()
print('Tempo impiegato nel calcolo della soluzione approssimata con FNO:', t2-t1)
    
# stampo a video output FNO
im = plt.imshow(out_test)
plt.colorbar(im)
plt.show()
   
# Valore assoluto della differenza tra sol esatta ed approssimata
diff = torch.abs(out_test - u_test)
# stampo a video differenza tra sol esatta ed approssimata
im = plt.imshow(diff)
plt.colorbar(im)
plt.show()

# errore in normal L2
loss = L2Loss()
err_L2 = loss(u_test.unsqueeze(0), out_test).item()
print('Errore in norma L2 =', err_L2)
# errore relativo in norma L2
loss_rel = L2relLoss()
err_rel_L2 = loss_rel(u_test.unsqueeze(0), out_test).item()
print('Errore relativo in norma L2 =', err_rel_L2)

# for p in model.parameters():
#     print(p.shape)


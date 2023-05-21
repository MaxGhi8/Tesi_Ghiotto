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
Path = "model_FFNO"

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
        # per fare un fully connected layer si puo' utilizzare un 2d convolutional
        # layer con opportuni paremetri. In particolare si deve utilizzare dei
        # convolutional layer con kernel_size=1, padding=0, di default viene
        # aggiunto anche il bias
        self.mlp1 = nn.Linear(in_channels, mid_channels)
        self.mlp2 = nn.Linear(mid_channels, out_channels)

    def forward(self, x):
        x = self.mlp1(x) # linear transformation
        x = activation(x) # activation function
        x = self.mlp2(x) # linear transformation
        return x

#########################################
# fourier layer
#########################################
class FourierLayer(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, fourier_weight, init_norm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.init_norm = init_norm

        # Meglio non usare parametri complessi.
        # https://github.com/pytorch/pytorch/issues/59998
        self.fourier_weight = fourier_weight
        # Se la matrice R_x e R_y per trasformare i Fourier weights
        # non la passo in input allora li definisco
        if not self.fourier_weight: 
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                # Xavier normalization
                if self.init_norm == 'xavier':
                    nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

    def forward(self, x):
        # x in input ha dim = (batch_size)*(n_x)*(n_y)*(d_v)
        x = x.permute(0, 3, 2, 1)
        # x.shape = (batch_size)*(d_v)*(n_x)*(n_y)
        B, I, M, N = x.shape
        
        ################
        #### dim. x ####
        ################
        x_ftx = torch.fft.rfft(x, dim = -2, norm = 'ortho')
        # x_ft.shape = (batch_size)*(d_v)*(n_x//2 + 1)*(n_y)
        
        # tensor di zeri con le stesse caratteristiche di x_ftx
        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        
        # Trasformazione lineare R
        out_ft[:, :, :self.modes_x, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.modes_x, :],
            torch.view_as_complex(self.fourier_weight[0]))
        
        # Trasformata di Fourier inversa
        xx = torch.fft.irfft(out_ft, n = M, dim = -2, norm = 'ortho')
        # x.shape = (batch_size)*(d_v)*(n_x)*(n_y)

        ################
        #### dim. y ####
        ################
        x_fty = torch.fft.rfft(x, dim = -1, norm = 'ortho')
        # x_ft.shape = (batch_size)*(d_v)*(n_x)*(n_y//2 + 1)
        
        # tensor di zeri con le stesse caratteristiche di x_fty
        out_ft = x_fty.new_zeros(B, I, M, N//2 + 1)
        
        # Trasformazione lineare R
        out_ft[:, :, :, :self.modes_y] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.modes_y],
            torch.view_as_complex(self.fourier_weight[1]))
        
        # Trasformata di Fourier inversa
        xy = torch.fft.irfft(out_ft, n = N, dim = -1, norm = 'ortho')
        # x.shape = (batch_size)*(d_v)*(n_x)*(n_y)

        #### Sommo le dimensioni
        x = xx + xy
        # Risistemo le dimensioni
        x  = x.permute(0, 2, 3, 1) # (batch_size)*(n_x)*(n_y)*(d_v)
        return x

#########################################
# FFNO for Darcy
#########################################  
class FFNO_Darcy(nn.Module):
    """ 
    Factorized Fourier Neural Operator per il problema di Darcy in dim. 2   
    """
    def __init__(self, d_a, d_v, d_u, L, modes_x, modes_y, share_weight, init_norm):
        """             
        d_a : int
            pari alla dimensione dello spazio in input
            
        d_v : int
            pari alla dimensione dello spazio nell'operatore di Fourier
        
        d_u : int
            pari alla dimensione dello spazio in output 
            
        L: int
            numero di Fourier operator da fare
            
        modes_x : int
            pari a k_{max, 1}
            
        modes_y : int
            pari a k_{max, 2}
            
        share_weight : bool
            Se è posto a True vuol dire che tutti i valori della trasformata
            di Fourier li trasformo con la stessa trasformazione lineare, se è
            posto a False allora usa una trasformazione lineare per ohni layer
        
        init_norm : string
            se è uguale a 'xavier' si usa la xavier normal inizialzation
            se è uguale a 'kaiming' si usa la kaiming normal inizialization #TODO
        """
        super().__init__()
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.share_weight = share_weight
        self.init_norm = init_norm
        self.padding = 8  # pad the domain if input is non-periodic
        
        #### Encoder
        self.p = nn.Linear(self.d_a, self.d_v) # input features is d_a=3: (a(x, y), x, y)
        
        #### Parametri per la trasformata di Fourier
        self.fourier_weight = None
        # Se share_weight è True allora definisco solo una trasformazione
        # lineare che passo in tutti i Fourier Layer senno' definisco una 
        # nuova trasformazione per ogni layer
        if self.share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y]:
                weight = torch.FloatTensor(d_v, d_v, n_modes, 2)
                param = nn.Parameter(weight)
                if self.init_norm == 'xavier':
                    nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.fourier_layer = nn.ModuleList([])
        self.ws1 = nn.ModuleList([])
        self.ws2 = nn.ModuleList([])
        for i in range(L):
            # Trasformazione lineare 1
            w1 = nn.Linear(self.d_v, self.d_v)
            self.ws1.append(w1)
            # Trasformazione lineare 2
            if i < self.L - 1:
                w2 = nn.Linear(self.d_v, self.d_v)
                self.ws2.append(w2)
            # Fourier Layer
            self.fourier_layer.append(FourierLayer(in_dim = d_v,
                                                     out_dim = d_v,
                                                     modes_x = modes_x,
                                                     modes_y = modes_y, 
                                                     fourier_weight = self.fourier_weight,
                                                     init_norm = self.init_norm))
            
        #### Decoder
        self.q = MLP(self.d_v, self.d_u, 4*self.d_u) # output features is d_u: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim = -1)  # (n_samples)*(n_x)*(n_y)*3
        
        # Applica P
        x = self.p(x) # shape == (n_samples)*(n_x)*(n_y)*(d_v)
        
        # Pad the domain
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])
        x = x.permute(0, 2, 3, 1)
        
        # Fourier Layers
        for i in range(self.L):
            z = self.fourier_layer[i](x)
            z = self.ws1[i](z)
            if i < self.L - 1: 
                z = activation(z)
                z = self.ws2[i](z)
                z = activation(z)
                x = x + z
            else:
                x = z

        # Unpad the domain
        x = x[..., :-self.padding, :-self.padding, :]
        
        # Applico q
        x = self.q(x)

        return x.squeeze(3) # Tolgo l'ultima dimensione
    
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


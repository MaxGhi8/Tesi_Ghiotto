"""
Implementazione di FNO per la risoluzione del problema di Darcy 2D.
"""
import matplotlib.pyplot as plt
import numpy as np
import operator
from functools import reduce
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# library for reading data
import scipy.io
# timer
from timeit import default_timer

#########################################
# nomi file da salvare
#########################################
name_log_dir = 'exp_test'
name_model = 'model_test'

#########################################
# valori di default
#########################################
# mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype

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
# reading data
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
            a = 0, mode = 'fan_in', nonlinearity = fun_act)
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
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2),
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
        self.p = nn.Linear(self.d_a, self.d_v) # input features is d_a=3: (a(x,y), x, y)
        
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
        self.q = nn.Linear(self.d_v, self.d_u) # output features is d_u: u(x,y)
        

    def forward(self, x):
        grid = self.get_grid(x.shape, mydevice)
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
        return x.squeeze() # tolgo l'ultima dim
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2] # n_sample, n_x, n_y
        # griglia x
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype = torch.float) # griglia uniforme
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1]) # adatto la dimensione
        # idem per la griglia y
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype = torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) #concateno lungo l'ultima dimensione

if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir )
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)
    
    #########################################
    # lettura dati e initial normalization
    ######################################### 
    s = 5 # parametro mesh: s=5 --> 85, s=1 --> 421
    
    # Training data
    ntrain = 1000 # training instances
    g = torch.Generator().manual_seed(1) # fisso seed
    idx_train = torch.randperm(1024, device = 'cpu', generator = g)[:ntrain]
    TrainDataPath = 'data/piececonst_r421_N1024_smooth1.mat'
    a_train, u_train = MatReader(TrainDataPath)
    a_train, u_train = a_train[idx_train, ::, ::], u_train[idx_train, ::, ::]    
    # Gaussian pointwise normalization
    a_normalizer = UnitGaussianNormalizer(a_train) #compute mean e std
    a_train = a_normalizer.encode(a_train) # normalize
    # extraction data
    a_train, u_train = a_train[:, ::s, ::s], u_train[:, ::s, ::s]
    _, nx_train, ny_train = a_train.size()
    
    # Test data
    ntest = 200 # test instances
    idx_test = torch.randperm(1024, device = 'cpu', generator = g)[:ntest]
    TestDataPath = 'data/piececonst_r421_N1024_smooth2.mat'
    a_test, u_test = MatReader(TestDataPath)
    # Gaussian pointwise normalization
    a_test = a_normalizer.encode(a_test) # normalize
    # extraction data
    a_test, u_test = a_test[idx_test, ::s, ::s], u_test[idx_test, ::s, ::s]
    _, nx_test, ny_test = a_test.size()

    # aggiungo una dimensione, utile in seguito per quando unisco la griglia
    a_train = a_train.reshape(ntrain, nx_train , ny_train, 1)
    a_test = a_test.reshape(ntest, nx_test, ny_test, 1)  
    
    print('Data loaded')  
    
    #########################################
    # iperparametri
    #########################################   
    # per il training     
    batch_size = 20
    learning_rate = 0.001
    epochs = 2
    iterations = epochs*(ntrain//batch_size)
    # per il modello
    d_a = 3 # dimensione spazio di input
    d_v = 32 # dimensione spazio nel Fourier operator
    d_u = 1 # dimensione dell'output
    L = 4
    modes = 12 # k_{max, j}
    BN = False # Batch normalization
    # per plot e tensorboard
    ep_step = 1
    # idx = [42] # lista di numeri a caso tra 0 e n_test-1
    idx = [7, 42, 93, 158] 
    n_idx = len(idx)
    plotting = True
    
    ################################################################
    # training, evaluation e plot
    ################################################################
    # Suddivisioni dei dati in batch
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test),
                                              batch_size = batch_size)
    
    # Inizializzazione del modello
    model = FNO_Darcy2d(d_a, d_v, d_u, L, modes, modes, BN)
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
    
    # dimezza il learning rate ogni 100 epoche
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    # Funzione da minimizzare
    myloss = L2relLoss()
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for a, u in train_loader:            
            a, u = a.to(mydevice), u.to(mydevice)
            
            optimizer.zero_grad() # azzero il gradiente all'inizio per i vari batch
            out = model.forward(a) 
            loss = myloss(out.view(batch_size, -1), u.view(batch_size, -1))
            loss.backward()
    
            optimizer.step()
            train_l2 += loss.item()
            
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad(): # per efficienza
            for a, u in test_loader:
                a, u = a.to(mydevice), u.to(mydevice)
    
                out = model.forward(a)      
                test_l2 += myloss(out.view(batch_size, -1), u.view(batch_size, -1)).item()
                
        train_l2 /= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        print('Epoch:', ep, 'Time:', t2-t1, 'Train_loss:', train_l2, 'Test_loss:', test_l2)
        writer.add_scalars('FNOdarcy2D', {'Train_loss': train_l2, 'Test_loss': test_l2}, ep)
        
        #########################################
        # plot dei dati alla fine ogni ep_step epoche
        #########################################
        if ep == 0:
            # Dato iniziale
            esempio_test = a_test[idx]
            
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
                im = ax[i].imshow(esempio_test[i])
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('coeff a(x)', fig, 0)
            
            # Soluzione esatta
            soluzione_test = u_test[idx]
            
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Soluzione esatta')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # x label
                ax[i].set(xlabel = 'x')
                # figura
                im = ax[i].imshow(soluzione_test[i])
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('soluzione', fig, 0)
                
        # Soluzione approssimata dalla FNO e differenza
        if ep % ep_step == 0:
            with torch.no_grad(): # no grad per efficienza
                out_test = model(esempio_test)   
                
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
                im = ax[i].imshow(out_test[i])
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('appro_sol', fig, ep)
            
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
                im = ax[i].imshow(diff[i])
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('diff', fig, ep)
    
    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    
    torch.save(model, name_model)   
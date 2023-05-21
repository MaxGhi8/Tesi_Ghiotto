"""
Implementazione di FFNO per la risoluzione del problema di Darcy 2D.
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
    def __init__(self, d_a, d_v, d_u, L, modes_x, modes_y, share_weight, 
                 init_norm, dropout, layer_norm):
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
            
        dropout: float
            numero da passare in input a nn.dropout, cioè è la probabilità di 
            fare dropout di un parametro
        
        layer_norm: bool
            Se posto True faccio layer normalization 
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
        self.dropout = dropout
        self.layer_norm = layer_norm
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
            # FFT e IFFT
            z = self.fourier_layer[i](x)
            # Prima trasformazione
            z = self.ws1[i](z)
            z = nn.Dropout(self.dropout)(z)
            z = activation(z)
            # Seconda trasformazione
            z = self.ws2[i](z)
            z = nn.Dropout(self.dropout)(z)
            if self.layer_norm:
                z = nn.LayerNorm(self.d_v)(z)
            x = x + z


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

if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir )
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)
    
    #########################################
    # lettura dati e initial normalization
    #########################################     
    s = 5 # parametro mesh: s=5 --> 85, s=1 --> 421
    # su una griglia da 85 i fourier modes sono 85//2+1, poi ne scarto metà
    # per la FFNO
    
    # Training data
    ntrain = 1000 # training instances
    g = torch.Generator().manual_seed(1) # fisso seed
    idx_train = torch.randperm(1024, device = 'cpu', generator = g)[:ntrain]
    TrainDataPath = 'data/piececonst_r421_N1024_smooth1.mat'
    a_train, u_train = MatReader(TrainDataPath)
    a_train, u_train = a_train[idx_train, ::, ::], u_train[idx_train, ::, ::]    
    # Gaussian pointwise normalization
    a_normalizer = UnitGaussianNormalizer(a_train) #compute mean e std
    a_train = a_normalizer.encode(a_train) # normalizzo
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
    modes = 20 # k_{max, j}, ne scarto metà
    share_weight = False
    init_norm = 'xavier'
    dropout = 0
    layer_norm = False
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
    model = FFNO_Darcy(d_a, d_v, d_u, L, modes, modes, share_weight,
                       init_norm, dropout, layer_norm)
    # model.to(mydevice)
    
    # conta del numero di parametri utilizzati
    par_tot = 0
    for p in model.parameters():
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape))
    print("Numero totale di parametri dell'operator network è:", par_tot)
    writer.add_text("Parametri", 'il numero totale di parametri è' + str(par_tot), 0)
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
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
        writer.add_scalars('FFNO_Darcy', {'Train_loss': train_l2, 'Test_loss': test_l2}, ep)
        
        #########################################
        # plot dei dati alla fine ogni ep_step epoche
        #########################################
        if ep == 0:
            # Dato iniziale
            esempio_test = a_test[idx]
            writer.add_graph(model, esempio_test, 0)
            
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
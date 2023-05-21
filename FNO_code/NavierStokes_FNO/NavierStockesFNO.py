"""
Implementazione di FNO per la risoluzione dei Navier Stokes (2d + time).
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
import h5py
# timer
from timeit import default_timer

#########################################
# nomi file da salvare
#########################################
name_log_dir = 'exp_test'
name_model = 'model_test'

init_weight_norm = 'kaiming'

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
    a = np.transpose(a, axes=range(len(a.shape) - 1, -1, -1))
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
        
        if init_weight_norm == 'kaiming':
            # Weights with kaiming initilization
            self.weights1 = torch.nn.init.kaiming_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                a=0, mode = 'fan_in', nonlinearity = fun_act)
            self.weights2 = torch.nn.init.kaiming_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                a = 0, mode = 'fan_in', nonlinearity = fun_act)  
        else:
            self.scale = (1 / (in_channels * out_channels))
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, 
                            out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, 
                            out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


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
# FNO per Navier Stockes 2D + time
#########################################
class FNO_NavierStockes_2dtime(nn.Module):
    """ 
    Fourier Neural Operator per il problema di NS 2d + time   
    input della rete è:
        (n_samples)*(n_x)*(n_y)*(T_in + 2)
    output della rete è:
        (n_samples)*(n_x)*(n_y)*(1)
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
        super(FNO_NavierStockes_2dtime, self).__init__()
        
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.BN = BN
        
        # Encoder
        self.p = nn.Linear(self.d_a, self.d_v) # input features is d_a
        
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
        x = torch.cat((x, grid), dim = -1) #concateno lungo l'ultima dimensione
        # ora x è un tensore di dimensione: (n_samples)*(n_x)*(n_y)*(12)
        
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
        return torch.cat((gridx, gridy), dim=-1).to(device) #concateno lungo l'ultima dimensione

if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir )
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)
    
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
    
    # Test data
    a_test, u_test = a[idx_test, :, :, :T_in], a[idx_test, :, :, T_in:T_in + T]  
    # Gaussian pointwise normalization
    a_test = a_normalizer.encode(a_test) # normalize
    # extraction data
    a_test, u_test = a_test[:, ::s, ::s, :], u_test[:, ::s, ::s, :]
    
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
    d_a = 12 # dimensione spazio di input
    # input_dim = 12: the solution of the previous 10 timesteps + 2 locations 
    # (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
    d_v = 20 # dimensione spazio nel Fourier operator
    d_u = 1 # dimensione dell'output
    L = 4
    modes = 12 # k_{max, j}
    BN = False # Batch normalization
    # per plot e tensorboard
    ep_step = 1
    idx = [42] # numero tra 0 e n_test-1 (come N ci metto i tempi)
    plotting = True # Toglierlo se non si vuole stampare a video ma salvare solo su tensorboard
    
    ################################################################
    # training, evaluation e plot
    ################################################################
    # Suddivisioni dei dati in batch
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train),
                                                batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test),
                                              batch_size = batch_size)
    
    # Inizializzazione del modello
    model = FNO_NavierStockes_2dtime(d_a, d_v, d_u, L, modes, modes, BN)
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
        train_l2_step = 0
        train_l2_full = 0
        for aa, uu in train_loader:
            loss_batch = 0
            aa, uu = aa.to(mydevice), uu.to(mydevice)
    
            for t in range(0, T, step):
                u = uu[..., t:t + step] # Previsione (t+step, T_in <= t+step <= T_in+T) corretta
                im = model(aa)
                loss_batch += myloss(im.reshape(batch_size, -1), u.reshape(batch_size, -1))
    
                if t == 0:
                    pred = im
                else:
                    # In prediction metto tutte le previsioni
                    pred = torch.cat((pred, im), -1)
                
                # Tolgo il primo e inserisco l'ultimo valore calcolato
                aa = torch.cat((aa[..., step:], im), -1) 
    
            train_l2_step += loss_batch.item()
            l2_full = myloss(pred.reshape(batch_size, -1), uu.reshape(batch_size, -1))
            train_l2_full += l2_full.item()
    
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
        scheduler.step()
    
        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for aa, uu in test_loader:
                loss_batch = 0
                aa, uu = aa.to(mydevice), uu.to(mydevice)
    
                for t in range(0, T, step):
                    u = uu[..., t:t + step]
                    im = model(aa)
                    loss_batch += myloss(im.reshape(batch_size, -1), u.reshape(batch_size, -1))
    
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
    
                    aa = torch.cat((aa[..., step:], im), -1)
    
                test_l2_step += loss_batch.item()
                test_l2_full += myloss(pred.reshape(batch_size, -1), uu.reshape(batch_size, -1)).item()
    
        train_l2_step = train_l2_step / ntrain / (T / step)
        train_l2_full /= ntrain
        test_l2_step = test_l2_step / ntest / (T / step)
        test_l2_full /= ntest
        t2 = default_timer()
        print('Epoch:', ep)
        print('Time:', t2 - t1)
        print('Train_loss_step:', train_l2_step, 'Train_loss_full:', train_l2_full)
        print('Test_loss_step:', test_l2_step, 'Test_loss_full:', test_l2_full)
        writer.add_scalars('NavierStockes_2dtime', 
                           {'Train_loss_step': train_l2_step,
                            'Train_loss_full': train_l2_full,
                            'Test_loss_step': test_l2_step,
                            'Test_loss_full': test_l2_full}, ep) 
        
        #########################################
        # plot dei dati alla fine ogni ep_step epoche
        #########################################
        if ep == 0:
            #### Dato iniziale
            esempio_test = a_test[idx]
            
            fig, ax = plt.subplots(2, 5, figsize = (18, 8))
            fig.suptitle('Tempi iniziali')
            for i in range(T_in):
                # Turn off tick labels
                ax[i//5, i%5].set_yticklabels([])
                ax[i//5, i%5].set_xticklabels([])
                # x label
                ax[i//5, i%5].set(xlabel = 'x')
                # y label
                if i%5 == 0:
                    ax[i//5, i%5].set(ylabel = 'y')
                # figura
                im = ax[i//5, i%5].imshow(esempio_test[..., i].squeeze())#i-esima osservzione
                fig.colorbar(im, ax = ax[i//5, i%5])
            if plotting:
                plt.show()
            writer.add_figure('Tempi iniziali (input)', fig, 0)

            #### Soluzione esatta
            soluzione_test = u_test[idx]
            
            fig, ax = plt.subplots(5, 8, figsize = (18, 12))
            fig.suptitle('Soluzioni per i tempi successivi')
            for i in range(T):
                # Turn off tick labels
                ax[i//8, i%8].set_yticklabels([])
                ax[i//8, i%8].set_xticklabels([])
                # x label
                ax[i//8, i%8].set(xlabel = 'x')
                # y label
                if i%8 == 0:
                    ax[i//8, i%8].set(ylabel = 'y')
                # figura
                im = ax[i//8, i%8].imshow(soluzione_test[..., i].squeeze())#i-esima osservzione
                fig.colorbar(im, ax = ax[i//8, i%8])
            if plotting:
                plt.show()
            writer.add_figure('Soluzioni per T tempi successivi', fig, 0)
                
        #### Soluzione approssimata dalla FNO e differenza
        if ep % ep_step == 0:
            with torch.no_grad(): # no grad per efficienza
                for t in range(0, T, step):
                    im = model(esempio_test)
                    if t == 0:
                        out_test = im
                    else:
                        # In out_test metto tutte le previsioni
                        out_test = torch.cat((out_test, im), -1)
                    # Tolgo il primo e inserisco l'ultimo valore calcolato
                    esempio_test = torch.cat((esempio_test[..., step:], im), -1)
            
            fig, ax = plt.subplots(5, 8, figsize = (18, 12))
            fig.suptitle('Soluzione approssimata')
            for i in range(T):
                # Turn off tick labels
                ax[i//8, i%8].set_yticklabels([])
                ax[i//8, i%8].set_xticklabels([])
                # x label
                ax[i//8, i%8].set(xlabel = 'x')
                # y label
                if i%8 == 0:
                    ax[i//8, i%8].set(ylabel = 'y')
                # figura
                im = ax[i//8, i%8].imshow(out_test[..., i].squeeze())#i-esima osservzione
                fig.colorbar(im, ax = ax[i//8, i%8])
            if plotting:
                plt.show()
            writer.add_figure('Soluzioni approssimate', fig, ep)
                
            # Valore assoluto della differenza tra sol esatta ed approssimata
            diff = torch.abs(out_test - soluzione_test)
            
            fig, ax = plt.subplots(5, 8, figsize = (18, 12))
            fig.suptitle('Differenza')
            for i in range(T):
                # Turn off tick labels
                ax[i//8, i%8].set_yticklabels([])
                ax[i//8, i%8].set_xticklabels([])
                # x label
                ax[i//8, i%8].set(xlabel = 'x')
                # y label
                if i%8 == 0:
                    ax[i//8, i%8].set(ylabel = 'y')
                # figura
                im = ax[i//8, i%8].imshow(diff[..., i].squeeze())#i-esima osservzione
                fig.colorbar(im, ax = ax[i//8, i%8])
            if plotting:
                plt.show()
            writer.add_figure('Differenza', fig, ep)
          
    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    
    torch.save(model, name_model)   
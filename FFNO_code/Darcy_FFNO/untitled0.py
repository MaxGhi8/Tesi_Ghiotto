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
    return F.tanh(x)

# per kaiming initial normalization
fun_act = 'relu' 

        
class FeedForward(nn.Module):
    def __init__(self, dim, factor, ff_weight_norm, n_layers, layer_norm, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.Dropout(dropout),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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


class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, m1, m2, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, dropout, s1, s2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.m1 = m1
        self.m2 = m2
        self.s1 = s1
        self.s2 = s2

        self.fourier_weight = fourier_weight
        # Can't use complex type yet. See https://github.com/pytorch/pytorch/issues/59998
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [m2, m1]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x, x_in=None, x_out=None, iphi=None, code=None, ff=True):
        x = self.forward_fourier(x, x_in, x_out, iphi, code)
        if ff:
            x = x.permute(0, 2, 3, 1)
            x = self.backcast_ff(x)
            x = x.permute(0, 3, 1, 2)
        return x

    def forward_fourier(self, x, x_in, x_out, iphi, code):
        # Compute the basis if needed
        if x_in is not None:
            basis_fft_x, basis_fft_y = self.get_fft_bases(x_in, iphi, code)
        if x_out is not None:
            basis_ifft_x, basis_ifft_y = self.get_ifft_bases(x_out, iphi, code)
        # basis_x.shape == [batch_size, n_points, m1, s2]
        # basis_y.shape == [batch_size, n_points, s1, m2]

        # # # Dimesion Y # # #
        if x_in is None:
            # x.shape == [batch_size, hidden_size, s1, s2]
            x_fty = torch.fft.rfft(x, dim=-1)
            # x_fty.shape == [batch_size, hidden_size, s1, s2 // 2 + 1]
        else:
            # x.shape == [batch_size, hidden_size, n_points]
            x_fty = torch.einsum("bcn,bnxy->bcxy", x + 0j, basis_fft_y)
            # x_fty.shape == [batch_size, hidden_size, s1, m2]

        B, H = x_fty.shape[:2]
        out_ft = x_fty.new_zeros(B, H, self.s1, self.s2 // 2 + 1)
        # out_ft.shape == [batch_size, hidden_size, s1, s2 // 2 + 1]

        out_ft[:, :, :, :self.m2] = torch.einsum(
            "bixy,ioy->boxy",
            x_fty[:, :, :, :self.m2],
            torch.view_as_complex(self.fourier_weight[0]))

        if x_out is None:
            xy = torch.fft.irfft(out_ft, n=self.s1, dim=-1)
            # xy.shape == [batch_size, in_dim, grid_size, grid_size]
        else:
            # out_ft.shape == [batch_size, hidden_size, s1, m2]

            xy = torch.einsum("bcxy,bnxy->bcn",
                              out_ft[:, :, :, :self.m2], basis_ifft_y).real
            # xy.shape == [batch_size, in_dim, n_points]

        # # # Dimesion X # # #
        if x_in is None:
            x_ftx = torch.fft.rfft(x, dim=-2)
            # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]
        else:
            # x.shape == [batch_size, hidden_size, n_points]
            x_ftx = torch.einsum("bcn,bnxy->bcxy", x + 0j, basis_fft_x)
            # x_ftx.shape == [batch_size, hidden_size, m1, s2]

        B, H = x_ftx.shape[:2]
        out_ft = x_ftx.new_zeros(B, H, self.s1 // 2 + 1, self.s2)
        # out_ft.shape == [batch_size, hidden_size, s1 // 2 + 1, s2]

        out_ft[:, :, :self.m1, :] = torch.einsum(
            "bixy,iox->boxy",
            x_ftx[:, :, :self.m1, :],
            torch.view_as_complex(self.fourier_weight[1]))

        if x_out is None:
            xx = torch.fft.irfft(out_ft, n=self.s2, dim=-2)
            # xx.shape == [batch_size, in_dim, grid_size, grid_size]
        else:
            xx = torch.einsum("bcxy,bnxy->bcn",
                              out_ft[:, :, :self.m1, :], basis_ifft_x).real
            # xy.shape == [batch_size, in_dim, n_points]

        # # Combining Dimensions # #
        x = xx + xy

        return x

    def get_fft_bases(self, mesh_coords, iphi, code):
        device = self.fourier_weight[0].device

        k_x1 = torch.arange(0, self.m1).to(device)
        k_x2 = torch.arange(0, self.m2).to(device)
        x = iphi(mesh_coords, code)  # [20, 972, 2]

        B, N, _ = x.shape
        K1 = torch.outer(x[..., 1].view(-1), k_x1)
        K1 = K1.reshape(B, N, self.m1)
        # K1.shape == [batch_size, n_points, m1]

        K2 = torch.outer(x[..., 0].view(-1), k_x2)
        K2 = K2.reshape(B, N, self.m2)
        # K2.shape == [batch_size, n_points, m2]

        basis_x = torch.exp(-1j * 2 * np.pi * K1).to(device)
        basis_x = basis_x.repeat([1, 1, 1, self.s2])
        # basis_x.shape == [batch_size, n_points, m1, s2]

        basis_y = torch.exp(-1j * 2 * np.pi * K2).to(device)
        basis_y = basis_y.repeat([1, 1, self.s1, 1])
        # basis_y.shape == [batch_size, n_points, s1, m2]

        return basis_x, basis_y

    def get_ifft_bases(self, mesh_coords, iphi, code):
        device = self.fourier_weight[0].device

        k_x1 = torch.arange(0, self.m1).to(device)
        k_x2 = torch.arange(0, self.m2).to(device)
        x = iphi(mesh_coords, code)  # [20, 972, 2]

        B, N, _ = x.shape
        K1 = torch.outer(x[..., 1].view(-1), k_x1)
        K1 = K1.reshape(B, N, self.m1)
        # K1.shape == [batch_size, n_points, m1]

        K2 = torch.outer(x[..., 0].view(-1), k_x2)
        K2 = K2.reshape(B, N, self.m2)
        # K2.shape == [batch_size, n_points, m2]

        basis_x = torch.exp(1j * 2 * np.pi * K1).to(device)
        basis_x = basis_x.repeat([1, 1, 1, self.s2])
        # basis_x.shape == [batch_size, n_points, m1, s2]

        basis_y = torch.exp(1j * 2 * np.pi * K2).to(device)
        basis_y = basis_y.repeat([1, 1, self.s1, 1])
        # basis_y.shape == [batch_size, n_points, s1, m2]

        return basis_x, basis_y


class FNOFullyFactorizedMesh2D(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels,
                 n_layers=4, is_mesh=True, s1=40, s2=40):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2
        self.n_layers = n_layers

        # input channel is 3: (a(x, y), x, y)
        self.fc0 = nn.Linear(in_channels, self.width)

        self.convs = nn.ModuleList([])
        self.ws = nn.ModuleList([])
        self.bs = nn.ModuleList([])

        # if factorized:
        #     self.fourier_weight = nn.ParameterList([])
        #     for _ in range(2):
        #         weight = torch.FloatTensor(width, width, modes1, 2)
        #         param = nn.Parameter(weight)
        #         nn.init.xavier_normal_(param, gain=1)
        #         self.fourier_weight.append(param)

        for i in range(self.n_layers + 1):
            conv = SpectralConv2d(in_dim=width,
                                  out_dim=width,
                                  m1=modes1,
                                  m2=modes2,
                                  backcast_ff=None,
                                  fourier_weight=None,
                                  factor=2,
                                  ff_weight_norm=True,
                                  n_ff_layers=2,
                                  layer_norm=False,
                                  dropout=0.0,
                                  s1=s1,
                                  s2=s2)
            self.convs.append(conv)

        self.bs.append(nn.Conv2d(2, self.width, 1))
        self.bs.append(nn.Conv1d(2, self.width, 1))

        for i in range(self.n_layers - 1):
            w = nn.Conv2d(self.width, self.width, 1)
            self.ws.append(w)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u.shape == [batch_size, n_points, 2] are the coords.
        # code.shape == [batch_size, 42] are the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in is None:
            x_in = u
        if self.is_mesh and x_out is None:
            x_out = u

        # grid is like the (x, y) coordinates of a unit square [0, 1]^2
        grid = self.get_grid([u.shape[0], self.s1, self.s2],
                             u.device).permute(0, 3, 1, 2)
        # grid.shape == [batch_size, 2, size_x, size_y] == [20, 2, 40, 40]
        # grid[:, 0, :, :] is the row index (y-coordinate)
        # grid[:, 1, :, :] is the column index (x-coordinate)

        # Projection to higher dimension
        u = self.fc0(u)
        u = u.permute(0, 2, 1)
        # u.shape == [batch_size, hidden_size, n_points]

        uc1 = self.convs[0](u, x_in=x_in, iphi=iphi,
                            code=code)  # [B, H, S1, S2]
        uc3 = self.bs[0](grid)  # [B, H, S1, S2]
        uc = uc1 + uc3  # [B, H, S1, S2]

        # [B, H, S1, S2]
        for i in range(1, self.n_layers):
            uc1 = self.convs[i](uc)
            uc3 = self.bs[0](grid)
            uc = uc + uc1 + uc3

        L = self.n_layers
        u = self.convs[L](uc, x_out=x_out, iphi=iphi, code=code, ff=False)
        # u.shape == [B, H, N]
        u3 = self.bs[-1](x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        # u.shape == [B, N, H]
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
    
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
    epochs = 20
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
    model = FNOFullyFactorizedMesh2D(modes, modes, d_v, in_channels, out_channels,
                     n_layers=4, is_mesh=True, s1=40, s2=40):
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
            scheduler.step()
            train_l2 += loss.item()
    
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
"""
geoFNO per problem of Navier Stockes on a pipe
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
name_log_dir = 'exp_test'
name_model = 'model_test'

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
Input_x = 'data/pipe/Pipe_X.npy'
Input_y = 'data/pipe/Pipe_Y.npy'
outout_sigma = 'data/pipe/Pipe_Q.npy'

g = torch.Generator().manual_seed(1) # fisso seed
ntrain = 1000 # training instances
ntest = 200 # test instances

t = 20 # tempi da discretizzare
s1 = 129 # numero di punti discretizzazione asse x
s2 = 129 # numero di punti discretizzazione asse y
r1 = 1 # prendo un punto ogni r1 nei punti di s1
r2 = 1 # prendo uno "strato" ogni r2
# aggiorno i valori di s1 e s2 in base alle scelte di r1 e r2
s1 = int(((s1 - 1) / r1) + 1)
s2 = int(((s2 - 1) / r2) + 1)

#########################################
# iperparametri del modello
#########################################   
#### per il training     
learning_rate = 0.001 # l.r.
epochs = 2 # number of epochs for training
batch_size = 20
iterations = epochs*(ntrain//batch_size)
# per il modello
d_a = 4 # dimensione spazio di input
d_v = 32 # dimensione spazio nel Fourier operator
d_u = 1 # dimensione dell'output
L = 4 
modes1 = 12 # k_{max, 1}
modes2 = 12 # k_{max, 1}
#### per plot e tensorboard
ep_step = 1 # ogni quante epoche salvare i plot su tensorboard
idx = [7, 42, 72, 175] # lista di numeri a caso tra 0 e t-1 (tempi che plotto)
n_idx = len(idx)
plotting = True # Per stampare a video

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
class PipeNS_geoFNO(nn.Module):
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2):
        super(PipeNS_geoFNO, self).__init__()
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

        input is the coordinate of the mesh and output is the x-component
        of the velocity on every point of the mesh
        input shape: (batchsize, x=129, y=129, c=2)
        poi all'input aggiungo i valori di x e y --> d_a = 4
        output shape: (batchsize, x=129, y=129, c=1)
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

if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir)
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)
    ################################################################
    # load data
    ################################################################
    inputX = np.load(Input_x)
    inputX = torch.tensor(inputX, dtype = torch.float)
    inputY = np.load(Input_y)
    inputY = torch.tensor(inputY, dtype = torch.float)
    Input = torch.stack([inputX, inputY], dim = -1)

    output = np.load(outout_sigma)[:, 0]
    output = torch.tensor(output, dtype = torch.float)

    a_train = Input[:ntrain, ::r1, ::r2][:, :s1, :s2]
    u_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    
    a_test = Input[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
    u_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train),
                                batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test),
                                batch_size=batch_size, shuffle=False)
    
    print("Data loaded")
    
    ################################################################
    # training, evaluation e plot
    ################################################################
    
    # Inizializzazione del modello
    model = PipeNS_geoFNO(d_a, d_v, d_u, L, modes1, modes2)
    # model.to(mydevice)
    
    # conta del numero di parametri utilizzati
    par_tot = 0
    for p in model.parameters():
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    print("Numero totale di parametri dell'operator network è:", par_tot)
    # salvo il numero di parametri su tensorboard
    writer.add_text("Parametri", 'il numero totale di parametri è' + str(par_tot), 0)
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)
    
    # Funzione da minimizzare
    if Loss == 'L2':
        myloss = L2relLoss()
    elif Loss == 'H1':
        myloss = H1relLoss()
        
    
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        for a, u in train_loader:
            a, u = a.to(mydevice), u.to(mydevice)
    
            optimizer.zero_grad() # azzero il gradiente
            # Calcolo la soluzione approssimata
            out = model(a)
            # Calcolo della loss e backward
            loss = myloss(out, u) # va bene sia per 'L2' che per 'H1'
            loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
    
        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        test_h1 = 0.0
        with torch.no_grad(): # for efficency
            for a, u in test_loader:
                a, u = a.to(mydevice), u.to(mydevice)
    
                out = model(a)
                test_l2 += L2relLoss()(out, u).item()
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
        size = 10 # dim dei punti per lo scatter
        
        #### Valori della mesh
        X = a_test[idx, : ,:, 0].squeeze()
        Y = a_test[idx, :, :, 1].squeeze()
        
        if ep == 0:
            #### Dato iniziale (mesh)
            in_test = a_test[idx]

            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Mesh iniziale')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                ax[i].set(xlabel = 'x')
                # figura del dato iniziale
                ax[i].scatter(X[i], Y[i], size)
            if plotting:
                plt.show()
            writer.add_figure('Mesh inziale', fig, 0)
            
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
                im = ax[i].pcolormesh(X[i], Y[i], sol_test[i], shading='gouraud')
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Soluzione esatta', fig, 0)
            
        if ep % ep_step == 0:
            #### Soluzione approssimata
            with torch.no_grad(): # efficiency
                out_test = model(in_test).squeeze()
            
            fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
            fig.suptitle('Soluzione approssimata')
            ax[0].set(ylabel = 'y')
            for i in range(n_idx):
                ax[i].set(xlabel = 'x')
                # Turn off tick labels
                ax[i].set_yticklabels([])
                ax[i].set_xticklabels([])
                # figura
                im = ax[i].pcolormesh(X[i], Y[i], out_test[i], shading='gouraud')
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Soluzione approssimata', fig, ep)
            
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
                im = ax[i].pcolormesh(X[i], Y[i], diff[i], shading='gouraud')
                fig.colorbar(im, ax = ax[i])
            if plotting:
                plt.show()
            writer.add_figure('Differenza', fig, ep)
  
    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    
    torch.save(model, name_model)  

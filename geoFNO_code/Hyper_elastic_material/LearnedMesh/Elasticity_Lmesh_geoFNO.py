"""
geoFNO for hyper-elastic material with Learned mesh
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
Path_Sigma = '../data/Meshes/Random_UnitCell_sigma_10.npy'
Path_XY = '../data/Meshes/Random_UnitCell_XY_10.npy'
Path_rr = '../data/Meshes/Random_UnitCell_rr_10.npy'

g = torch.Generator().manual_seed(1) # fisso seed
ntrain = 1000 # training instances
ntest = 200 # test instances

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
modes = 12 # k_{max, 1}
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

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1 + K2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, d_v, in_channels, out_channels, is_mesh=True, s1=40, s2=40):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.d_v = d_v
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.d_v)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.d_v, self.d_v, self.modes1, self.modes2, s1, s2)
        self.conv1 = SpectralConv2d(self.d_v, self.d_v, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.d_v, self.d_v, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.d_v, self.d_v, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.d_v, self.d_v, self.modes1, self.modes2, s1, s2)
        self.w1 = nn.Conv2d(self.d_v, self.d_v, 1)
        self.w2 = nn.Conv2d(self.d_v, self.d_v, 1)
        self.w3 = nn.Conv2d(self.d_v, self.d_v, 1)
        self.b0 = nn.Conv2d(2, self.d_v, 1)
        self.b1 = nn.Conv2d(2, self.d_v, 1)
        self.b2 = nn.Conv2d(2, self.d_v, 1)
        self.b3 = nn.Conv2d(2, self.d_v, 1)
        self.b4 = nn.Conv1d(2, self.d_v, 1)

        self.fc1 = nn.Linear(self.d_v, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class IPHI(nn.Module):
    def __init__(self, d_v=32):
        super(IPHI, self).__init__()

        """
        inverse phi: x -> xi
        """
        self.d_v = d_v
        self.fc0 = nn.Linear(4, self.d_v)
        self.fc_code = nn.Linear(42, self.d_v)
        self.fc_no_code = nn.Linear(3*self.d_v, 4*self.d_v)
        self.fc1 = nn.Linear(4*self.d_v, 4*self.d_v)
        self.fc2 = nn.Linear(4*self.d_v, 4*self.d_v)
        self.fc3 = nn.Linear(4*self.d_v, 4*self.d_v)
        self.fc4 = nn.Linear(4*self.d_v, 2)
        self.activation = torch.tanh
        self.center = torch.tensor([0.0001,0.0001]).reshape(1,1,2)

        self.B = np.pi*torch.pow(2, torch.arange(0, self.d_v//4, dtype=torch.float)).reshape(1,1,1,self.d_v//4)


    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # code (batch, N_features)

        # some feature engineering
        angle = torch.atan2(x[:,:,1] - self.center[:,:, 1], x[:,:,0] - self.center[:,:, 0])
        radius = torch.norm(x - self.center, dim=-1, p=2)
        xd = torch.stack([x[:,:,0], x[:,:,1], angle, radius], dim=-1)

        # sin features from NeRF
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(self.B * xd.view(b,n,d,1)).view(b,n,d*self.d_v//4)
        x_cos = torch.cos(self.B * xd.view(b,n,d,1)).view(b,n,d*self.d_v//4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b,n,3*self.d_v)

        if code!= None:
            cd = self.fc_code(code)
            cd = cd.unsqueeze(1).repeat(1,xd.shape[1],1)
            xd = torch.cat([cd,xd],dim=-1)
        else:
            xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


if __name__ == '__main__':
    # Per salvare i dati
    writer = SummaryWriter(log_dir = name_log_dir)
    # 'cuda' se è disponibile la GPU, sennò è 'cpu'
    print('Device disponibile:', mydevice)

    ################################################################
    # load data and data normalization
    ################################################################

    input_rr = np.load(Path_rr)
    input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0)
    train_rr = input_rr[:ntrain]
    test_rr = input_rr[-ntest:]
    
    input_s = np.load(Path_Sigma)
    input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    
    input_xy = np.load(Path_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]
        
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_rr, train_s, train_xy), 
                        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_rr, test_s, test_xy),
                         batch_size=batch_size, shuffle=False)
    
    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, d_v, in_channels=2, out_channels=1)
    model_iphi = IPHI()
    # model_iphi = FNO2d(modes, modes, d_v, in_channels=2, out_channels=2, is_skip=True)
    # Parametri totali
    params = list(model.parameters()) + list(model_iphi.parameters())
    
    # conta del numero di parametri utilizzati
    par_tot = 0
    for p in params:
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    print("Numero totale di parametri dell'operator network è:", par_tot)
    # salvo il numero di parametri su tensorboard
    writer.add_text("Parametri", 'il numero totale di parametri è' + str(par_tot), 0)
    
    # Adam optimizer
    
    optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay = 1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = iterations)
    
    # Funzione da minimizzare
    if Loss == 'L2':
        myloss = L2relLoss()
    elif Loss == 'H1':
        myloss = H1relLoss()
        
    N_sample = 1000
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_loss = 0
        train_reg = 0
        for rr, sigma, mesh in train_loader:
            rr, sigma, mesh = rr.to(mydevice), sigma.to(mydevice), mesh.to(mydevice)
            samples_x = torch.rand(batch_size, N_sample, 2) * 3 -1
    
            optimizer.zero_grad()
            out = model(mesh, code=rr, iphi=model_iphi)
            samples_xi = model_iphi(samples_x, code=rr)
    
            loss_data = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
            loss_reg = myloss(samples_xi, samples_x)
            loss = loss_data + 0.000 * loss_reg
            loss.backward()
    
            optimizer.step()
            train_loss += loss_data.item()
            train_reg += loss_reg.item()
    
        scheduler.step()
    
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for rr, sigma, mesh in test_loader:
                rr, sigma, mesh = rr.to(mydevice), sigma.to(mydevice), mesh.to(mydevice)
                # out = model(mesh, iphi=model_iphi)
                out = model(mesh, code=rr, iphi=model_iphi)
                test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()
    
        train_loss /= ntrain
        train_reg /= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        print('Epoch:', ep,
              'Time:', t2-t1,
              'Train_loss:', train_loss,
              'Test_loss_l2:', test_l2)
        writer.add_scalars('FNOplus_darcy2D', {'Train_loss': train_loss,
                                               'Test_loss_l2': test_l2}, ep)
        
        if ep%1==0:
            XY = mesh[-1].squeeze().detach().cpu().numpy()
            truth = sigma[-1].squeeze().detach().cpu().numpy()
            pred = out[-1].squeeze().detach().cpu().numpy()
    
            lims = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
            ax[0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
            ax[1].scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, **lims)
            ax[2].scatter(XY[:, 0], XY[:, 1], 100, truth - pred, edgecolor='w', lw=0.1, **lims)
            fig.show()
    
    #     #########################################
    #     # plot dei dati alla fine ogni ep_step epoche
    #     #########################################
    #     size = 100 # dim dei punti per lo scatter

    #     if ep == 0:
    #         #### Dato iniziale (mesh)
    #         in_test = test_xy[idx]
    #         #### Valori della mesh
    #         X = in_test[:, :, 0].numpy()
    #         Y = in_test[:, :, 1].numpy()

    #         fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    #         fig.suptitle('Mesh iniziale')
    #         ax[0].set(ylabel = 'y')
    #         for i in range(n_idx):
    #             ax[i].set(xlabel = 'x')
    #             # figura del dato iniziale
    #             ax[i].scatter(X[i], Y[i], size)            
    #         if plotting:
    #             plt.show()
    #         writer.add_figure('Mesh iniziale', fig, 0)
            
    #         #### Soluzione esatta
    #         sol_test = test_s[idx]
    #         fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    #         fig.suptitle('Soluzione esatta')
    #         ax[0].set(ylabel = 'y')
    #         for i in range(n_idx):
    #             ax[i].set(xlabel = 'x')
    #             # Turn off tick labels
    #             ax[i].set_yticklabels([])
    #             ax[i].set_xticklabels([])
    #             # per la colormap (vale per tutti i plot)
    #             lims = dict(cmap='RdBu_r', vmin = sol_test[i].min(), vmax = sol_test[i].max())
    #             # figura
    #             sol_test_i = sol_test[i]
    #             im = ax[i].scatter(X[i], Y[i], size, sol_test_i, edgecolor='w', lw=0.1, **lims)
    #             fig.colorbar(im, ax = ax[i])
    #         if plotting:
    #             plt.show()
    #         writer.add_figure('Soluzione esatta', fig, 0)
            
    #     if ep % ep_step == 0:
    #         #### Soluzione approssimata
    #         with torch.no_grad(): # efficiency
    #             out_test = model(in_test).squeeze()
            
    #         fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    #         fig.suptitle('Soluzione approssimata')
    #         ax[0].set(ylabel = 'y')
    #         for i in range(n_idx):
    #             ax[i].set(xlabel = 'x')
    #             # Turn off tick labels
    #             ax[i].set_yticklabels([])
    #             ax[i].set_xticklabels([])
    #             # per la colormap (vale per tutti i plot)
    #             lims = dict(cmap='RdBu_r', vmin = sol_test[i].min(), vmax = sol_test[i].max())
    #             # figura
    #             im = ax[i].scatter(X[i], Y[i], size, out_test[i], edgecolor='w', lw=0.1, **lims)
    #             fig.colorbar(im, ax = ax[i])
    #         if plotting:
    #             plt.show()
    #         writer.add_figure('Soluzione approssimata', fig, ep)
            
    #         #### Differenza
    #         diff = np.abs(sol_test - out_test)
    #         fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    #         fig.suptitle('Differenza')
    #         ax[0].set(ylabel = 'y')
    #         for i in range(n_idx):
    #             ax[i].set(xlabel = 'x')
    #             # Turn off tick labels
    #             ax[i].set_yticklabels([])
    #             ax[i].set_xticklabels([])
    #             # per la colormap (vale per tutti i plot)
    #             lims = dict(cmap='RdBu_r', vmin = sol_test[i].min(), vmax = sol_test[i].max())
    #             # figura
    #             im = ax[i].scatter(X[i], Y[i], size, np.abs(diff[i]), edgecolor='w', lw=0.1, **lims)
    #             fig.colorbar(im, ax = ax[i])
    #         if plotting:
    #             plt.show()
    #         writer.add_figure('Differenza', fig, ep)
  
    # writer.flush() # per salvare i dati finali
    # writer.close() # chiusura writer tensorboard
    
    # torch.save(model, name_model)  
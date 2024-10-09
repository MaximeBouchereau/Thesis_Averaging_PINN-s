# noinspection PyInterpreter
import sys
from time import sleep
import warnings

import scipy.optimize

warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
# print("No Warning Shown")

import torch
import torch.optim as optim
import torch.nn as nn
import copy

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from itertools import product
import statistics

# import autograd.numpy as np
# from autograd import grad
# from autograd import jacobian
# from autograd import hessian
# from torchdiffeq import odeint

import time
import datetime
from datetime import datetime as dtime

import webcolors

# Using of Neural Networks in order to integrate highly oscillatory ODE's


# Maths parameters [adjust]

dyn_syst = "VDP"             # Choice between "Logistic", "VDP" (Van der Pol oscillator), "I_Pendulum" or "Henon-Heiles
step_h = [0.01, 0.01]         # Interval where time step is selected
step_eps = [0.01, 1]         # Interval where high oscillation parameter is selected
T_simul = 1                 # Time for ODE's simulation
h_simul = 0.001                # Time step used for ODE's simulation
eps_simul = 1e-3              # High oscillation parameter used for ODE's simulation

# AI parameters [adjust]

K_data = 100                 # Quantity of data
R = 2                          # Amplitude of data in space (i.e. space data will be selected in the box [-R,R]^d)
p_train = 0.8                  # Proportion of data for training
HL = 2                         # Hidden layers per MLP for the first Neural Network
zeta = 200                     # Neurons per hidden layer of the first Neural Network
alpha = 2e-4                   # Learning rate for gradient descent
Lambda = 1e-9                  # Weight decay
BS = 200                        # Batch size (for mini-batching) for first training
N_epochs = 200                 # Epochs
N_epochs_print = 20            # Epochs between two prints of the Loss value


print(150 * "-")
print("Learning of solution of highly oscillatory ODE's")
print(150 * "-")

print("   ")
print(150 * "-")
print("Parameters:")
print(150 * "-")
print('    # Maths parameters:')
print("        - Dynamical system:", dyn_syst)
print("        - Interval where time step is selected:", step_h)
print("        - Interval where high oscillation parameter is selected:", step_eps)
print("        - Time for ODE's simulation:", T_simul)
print("        - Time step used for ODE's simulation:", h_simul)
print("        - High oscillation parameter used for ODE's simulation:", eps_simul)
print("    # AI parameters:")
print("        - Data's number for first training:", K_data)
print("        - Amplitude of data in space:", R)
print("        - Proportion of data for training:", format(p_train, '.1%'))
print("        - Hidden layers per MLP's for the first Neural Network:", HL)
print("        - Neurons on each hidden layer for the first Neural Network:", zeta)
print("        - Learning rate:", format(alpha, '.2e'))
print("        - Weight decay:", format(Lambda, '.2e'))
print("        - Batch size (mini-batching) for first training:", BS)
print("        - Epochs for first training:", N_epochs)
print("        - Epochs between two prints of the Loss value for first training:", N_epochs_print)

# Dimension of the problem

if dyn_syst == "Logistic":
    d = 1
if dyn_syst == "VDP" or dyn_syst == "I_Pendulum":
    d = 2
if dyn_syst == "Henon-Heiles":
    d = 4

def y0start(dyn_syst):
    """Gives the initial data (vector) for ODE's integration and study of trajectories"""
    if d == 1:
        return np.array([1.1])
    if d == 2:
        if dyn_syst == "Linear":
            return np.array([1.0,0.0])
        if dyn_syst == "VDP":
            return np.array([0.5,0.5])
        if dyn_syst == "I_Pendulum":
            return np.array([1.0,0.0])
    if d == 4:
        return np.array([0,0,1.5,1.5])

class NA:
    """Numerical analysis class, contains:
     - Function which describes the dynamical system
     - Function which creates data for training/test
     - ODE Integrator"""

    def f(tau,y):
        """Describes the dynamics of the studied ODE.
        Inputs:
         - tau : Float - Time variable
         - y : Array of shape (d,) - Space variable
         Returns an array of shape (d,)"""
        y = np.reshape(np.array(y) , (d,))
        if dyn_syst == "Logistic":
            z = np.zeros(d)
            z[0] = y[0]*(1-y[0]) + np.sin(tau)
            return z
        if dyn_syst == "Linear":
            return np.array([ ( -1 + np.cos(tau) )*y[0] - np.sin(tau)*y[1]  , (1+np.sin(tau))*y[0] + np.cos(tau)*y[1] ])
        if dyn_syst == "VDP":
            z = np.zeros(d)
            z[0] = - np.sin(tau)*(1/4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            z[1] = np.cos(tau)*(1/4 - (np.cos(tau) * y[0] + np.sin(tau) * y[1]) ** 2) * (-np.sin(tau) * y[0] + np.cos(tau) * y[1])
            return z
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0], y[1]
            z = np.zeros_like(y)
            z[0] = y2 + np.sin(tau) * np.sin(y1)
            z[1] = np.sin(y1) - (1 / 2) * np.sin(tau) ** 2 * np.sin(2 * y1) - np.sin(tau) * np.cos(y1) * y2
            return z
        if dyn_syst == "Henon-Heiles":
            z = np.zeros(d)
            z[0] = 2 * np.sin(tau) * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) * y[1]
            z[1] = y[3]
            z[2] = -2 * np.cos(tau) * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) * y[1]
            z[3] = -1 * (y[0] * np.cos(tau) + y[2] * np.sin(tau)) ** 2 + (3 / 2) * y[1] ** 2 - y[1]
            return z


    def f_NN(tau,y):
        """Returns the synamics of the ODE, usefulm for tensor inputs and Neural Networks.
        Inputs:
         - tau : Tensor of shape (1,n) - Time variable
         - y : Tensor of shape (d,n) - Space variable
         Returns a tensor of shape (d,n)"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)
        if dyn_syst == "Logistic":
            z[0,:] = y[0,:]*(1-y[0,:]) + torch.sin(tau)
            return z
        if dyn_syst == "VDP":
            z[0,:] = - torch.sin(tau)*(1/4 - (torch.cos(tau) * y[0,:] + torch.sin(tau) * y[1,:]) ** 2) * (-torch.sin(tau) * y[0,:] + torch.cos(tau) * y[1,:])
            z[1,:] = torch.cos(tau)*(1/4 - (torch.cos(tau) * y[0,:] + torch.sin(tau) * y[1,:]) ** 2) * (-torch.sin(tau) * y[0,:] + torch.cos(tau) * y[1,:])
            return z
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0,:], y[1,:]
            z[0] = y2 + torch.sin(tau) * torch.sin(y1)
            z[1] = torch.sin(y1) - (1 / 2) * torch.sin(tau) ** 2 * torch.sin(2 * y1) - torch.sin(tau) * torch.cos(y1) * y2
            return z

        if dyn_syst == "Henon-Heiles":
            y1, y2, y3, y4 = y[0, :], y[1, :], y[2, :], y[3, :]
            z[0, :] = 2 * torch.sin(tau) * (y1 * torch.cos(tau) + y3 * torch.sin(tau)) * y2
            z[1, :] = y4
            z[2, :] = -2 * torch.cos(tau) * (y1 * torch.cos(tau) + y3 * torch.sin(tau)) * y2
            z[3, :] = -1 * (y1 * torch.cos(tau) + y3 * torch.sin(tau)) ** 2 + (3 / 2) * y2 ** 2 - y2
            return z

    def f_av(t,y):
        """Gives the averaged vector field of the studied ODE
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable"""
        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)
        if dyn_syst == "Linear":
            z[0,:] = -y[0,:]
            z[1,:] = y[0,:]
        if dyn_syst == "VDP":
            z[0, :] = (-1 / 8) * (y[0, :] ** 2 + y[1, :] ** 2 - 1) * y[0, :]
            z[1, :] = (-1 / 8) * (y[0, :] ** 2 + y[1, :] ** 2 - 1) * y[1, :]
        if dyn_syst == "I_Pendulum":
            y1, y2 = y[0, :], y[1, :]
            z[0, :] = y2
            z[1, :] = torch.sin(y1) - (1 / 4) * torch.sin(2 * y1)
        if dyn_syst == "Logistic":
            z[0, :] = y[0, :]*(1-y[0, :])
        return z

    def F_av_1(t,y,epsilon):
        """Gives the averaged Field F^{epsilon} at order 1 (used for integration at stroboscopic times)
         - t : Float - Time variable
         - y : Array of shape (d,) - Space variable
         - epsilon : Float - High oscillation parameter
         CAUTION: Only for the Linear system and Logistic equation!!!"""
        y = np.reshape(np.array(y), (d,))
        if dyn_syst == "Linear":
            return np.array([(-1 + epsilon) * y[0] + (epsilon-epsilon**2) * y[1] , (1+epsilon) * y[0] - epsilon*(1+epsilon) * y[1]])/(1+epsilon**2)
        if dyn_syst == "Logistic":
            return np.array([ y[0]*(1-y[0])+ epsilon*(1-2*y[0])  - (3/2)*epsilon**2 ])

    def f_av_NN(y):
        """Returns the averaged vector field associated to the corresponding dynamical system, useful for neural network
        Inputs:
        - y: Tensor of shape (d,1) - Space variable"""

        nb_coeff = 1
        for s in y.shape:
            nb_coeff = nb_coeff * s
        y = torch.tensor(y).reshape(d, int(nb_coeff / d))
        z = torch.zeros_like(y)

        if dyn_syst == "Linear":
            z[0, :] = -y[0, :]
            z[1, :] = y[0, :]
        if dyn_syst == "VDP":
            z[0, :] = (-1/8)*(y[0, :]**2+y[1, :]**2-1)*y[0, :]
            z[1, :] = (-1/8)*(y[0, :]**2+y[1, :]**2-1)*y[1, :]
        if dyn_syst == "Logistic":
            z[0, :] = y[0, :]*(1-y[0, :])

        return z

    def ODE_Solve(ti , tf , h  , epsilon , Yinit):
        """Solves the highly oscillatory ODE:
               y' = f(t/epsilon , y)
        by using a DOP853 method which approximates the exact flow, from time t_0 to time t_0 + h.
         - ti : Float - Starting time of integration
         - tf : Float - Ending time of integration
         - h: Float - Duration of integration
         - epsilon : Float - Parameter of high oscillation
         - Yinit : Array of shape (d,) -Initial condition"""

        Yinit = np.reshape( np.array(Yinit) , (d,) )

        def ff(t, y):
            """Describes the dynamics of the studied ODE with high oscillations
             - t : Float - Time variable
             - y : Array of shape (d,) - Space variable
             Returns an array of shape (d,)"""
            y = np.reshape(np.array(y), (d,))
            return NA.f(t/epsilon , y)

        S = solve_ivp(fun = ff , t_span = (ti , tf + 2*h) , y0 = Yinit , method = "DOP853" , atol = 1e-8 , rtol = 1e-8 , t_eval = np.arange(ti , tf + h , h) )
        return S.y

class DataCreate:
    """Data class, for Data creation, contains function for Data creation"""

    def Data(K):
     """Function for Data creation, computes solutions at times t = 2*pi*epsilon where epsilon is randommly
     selected in the interval eps for the ODE
            y' = f( t/epsilon , y)
     - K : Integer - Number of data created
     Returns a tuple containing:
     - Initial data for training and test
     - Solutions at times t = 2*pi*epsilon for training and test
     - Epsilons for training and test
     All the componants of the tuple are tensors of shape (d,K0)/(d,K-K0) for solutions and shape (1,K0)/(1,K-K0)
     for epsilons, where K0 is the number of data used for training."""

     print(" ")
     print(150 * "-")
     print("Data creation...")
     print(150 * "-")

     start_time_data = time.time()

     Y0 , Y1 = np.random.uniform(low = -R , high = R , size = (d,K)) , np.zeros((d,K))
     T0 = np.random.uniform(low = 0 , high = T_simul , size = (1,K))
     #H = np.exp(np.random.uniform(low = np.log(step_h[0]) , high = np.log(step_h[1]) , size = (1,K)))
     #EPS = np.exp(np.random.uniform(low = np.log(step_eps[0]) , high = np.log(step_eps[1]) , size = (1,K)))
     H = h_simul * np.ones_like(T0)
     EPS = eps_simul * np.ones_like(T0)

     if dyn_syst == "Logistic":
         Y0 = np.abs(Y0)

     pow = max([int(np.log10(K) - 1), 3])
     pow = min([pow, 6])

     for k in range(K):
         end_time_data = start_time_data + (K / (k + 1)) * (time.time() - start_time_data)
         end_time_data = datetime.datetime.fromtimestamp(int(end_time_data)).strftime(' %Y-%m-%d %H:%M:%S')
         print(" Loading :  {} % \r".format(str(int(10 ** (pow) * (k + 1) / K) / 10 ** (pow - 2)).rjust(3)), " Estimated time for ending : " + end_time_data, " - ", end="")
         Y1[:,k] = NA.ODE_Solve(ti = 0 , tf = T0[0,k] , h = T0[0,k] , epsilon = EPS[0,k] , Yinit = Y0[:,k])[:,1]

     K0 = int(p_train*K)
     Y0_train = torch.tensor(Y0[:, 0:K0])
     Y0_test = torch.tensor(Y0[:, K0:K])
     Y1_train = torch.tensor(Y1[:, 0:K0])
     Y1_test = torch.tensor(Y1[:, K0:K])
     T0_train = torch.tensor(T0[:, 0:K0])
     T0_test = torch.tensor(T0[:, K0:K])
     H_train = torch.tensor(H[:, 0:K0])
     H_test = torch.tensor(H[:, K0:K])
     EPS_train = torch.tensor(EPS[:, 0:K0])
     EPS_test = torch.tensor(EPS[:, K0:K])

     print("Computation time for data creation (h:min:s):",
           str(datetime.timedelta(seconds=int(time.time() - start_time_data))))
     return (Y0_train , Y0_test , Y1_train , Y1_test , T0_train , T0_test , H_train , H_test , EPS_train , EPS_test)

class NN(nn.Module, NA):
    def __init__(self):
        super().__init__()
        #self.Psi = nn.ModuleList([nn.Linear(1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.Psi = nn.ModuleList([nn.Linear(d + 1, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])
        self.Phi_Pert = nn.ModuleList([nn.Linear(d + 2, zeta), nn.Tanh()] + (HL - 1) * [nn.Linear(zeta, zeta), nn.Tanh()] + [nn.Linear(zeta, d, bias=True)])

    def forward(self, t, y):
        """Structured Neural Network.
        Inputs:
         - y: Tensor of shape (d,n) - space variable
         - t: Tensor of shape (1,n) - Time variable"""

        y = y.T
        y = y.float()
        t = torch.tensor(t).T
        #h = torch.tensor(h).T
        #eps = torch.tensor(eps).T

        tau = t/eps_simul

        #x_Psi = t
        x_Psi = torch.cat((y , t), dim=1)

        for i, module in enumerate(self.Psi):
            x_Psi = module(x_Psi)

        #x_Psi = y + h_simul*NA.f_av_NN(y.T).T + eps_simul*h_simul*x_Psi

        x_Phi_Pert = torch.cat((torch.cos(tau), torch.sin(tau), x_Psi), dim=1)
        x_Phi_Pert_0 = torch.cat((torch.ones_like(tau), torch.zeros_like(tau), x_Psi), dim=1)

        for i, module in enumerate(self.Phi_Pert):
            x_Phi_Pert = module(x_Phi_Pert)
            x_Phi_Pert_0 = module(x_Phi_Pert_0)

        x_NN = x_Psi + eps_simul*(x_Phi_Pert - x_Phi_Pert_0)

        return (x_NN).T

class Train(NN, NA):
    """Training of the neural network, depends on the numerical method chosen
    Choice of the numerical method:
        - Forward Euler
        - MidPoint"""

    def Loss(self, Y0, TN, model):
        """Computes the Loss function with a PINN associated to the studied ODE.
        Inputs:
        - Y0: Tensor of shape (d,n) - Space variable
        - TN: Tensor of shape (1,n) - Time variable
        - model: Neural network which will be optimized
        Computes the L2  norm of the difference between derivative of the learned flow and vector field applied to the learned flow, as a PINN.
        => Returns a tensor of shape (1,1)"""
        Y0 = torch.tensor(Y0, dtype=torch.float32)
        Y0.requires_grad = True
        TN = torch.tensor(TN, dtype=torch.float32)
        TN.requires_grad = True
        ONE = torch.ones_like(TN)
        eta = 1e-5 # Small parameter to approximate derivative via finite difference method.

        d_Y_theta = (model(TN + eta*ONE , Y0) - model(TN - eta*ONE , Y0))/(2*eta)
        F_Y_theta = NA.f_NN(TN/eps_simul , model(TN , Y0))

        #d_Y_theta_0 = (model(0*TN + eta * ONE, Y0) - model(0*TN - eta * ONE, Y0)) / (2 * eta)
        #F_Y_theta_0 = NA.f_NN(0*TN / eps_simul, model(0*TN, Y0))

        #print(((d_Y_theta_0 - F_Y_theta_0)**2).mean())
        loss_ODE = ((d_Y_theta - F_Y_theta).abs() ** 2).mean()
        loss_IC = ((Y0 - model(0*TN , Y0)) ** 2).mean()

        loss = 0.1*loss_ODE + loss_IC
        #loss = loss_ODE + loss_IC
        #loss = (loss_ODE + loss_IC + (loss_ODE - loss_IC).abs())/2
        #loss = ((Y0 - model(0*TN , Y0)) ** 2).mean()

        return loss

    def train(self, model):
        """Makes the training on the data
        Inputs:
        - model: Neural network which will be optimized
        - Data: Tuple of tensors - Set of data created
        => Returns the lists Loss_train and Loss_test of the values of the Loss respectively for training and test,
        and best_model, which is the best apporoximation of the modified field computed"""

        start_time_train = time.time()

        print(" ")
        print(150 * "-")
        print("First training...")
        print(150 * "-")

        Y0 = 2*R*torch.rand(d,K_data) - R
        #Y0 = torch.tensor(y0start(dyn_syst)).reshape(d,1).float()@torch.ones(1,K_data)
        TN = T_simul*torch.rand(1,K_data)
        #TN = torch.tensor(np.random.exponential(scale = 0.1 , size = (1,K_data)))

        K0 = int(p_train*K_data) # Proportion of data used for train

        Y0_train = Y0[:,0:K0]
        Y0_test = Y0[:,K0:K_data]
        TN_train = TN[:,0:K0]
        TN_test = TN[:,K0:K_data]

        #TN_train[:,0:K0//10] = torch.zeros_like(TN_train[:,0:K0//10])

        optimizer = optim.AdamW(model.parameters(), lr=alpha, betas=(0.9, 0.999), eps=1e-8, weight_decay=Lambda,amsgrad=True)  # Algorithm AdamW
        best_model, best_loss_train, best_loss_test = model, np.infty, np.infty  # Selects the best minimizer of the Loss function
        Loss_train = [] # list for loss_train values
        Loss_test = []  # List for loss_test values

        for epoch in range(N_epochs + 1):
            for ixs in torch.split(torch.arange(Y0_train.shape[1]), BS):
                optimizer.zero_grad()
                model.train()
                Y0_batch = Y0_train[:, ixs]
                TN_batch = TN_train[:, ixs]
                loss_train = self.Loss(Y0_batch, TN_batch, model)
                loss_train.backward()
                optimizer.step()  # Optimizer passes to the next epoch for gradient descent

            loss_test = self.Loss(Y0_test, TN_test, model)

            if loss_train < best_loss_train:
                best_loss_train = loss_train
                best_loss_test = loss_test
                best_model = copy.deepcopy(model)
                # best_model = model

            Loss_train.append(loss_train.item())
            Loss_test.append(loss_test.item())

            if epoch % N_epochs_print == 0:  # Print of Loss values (one print each N_epochs_print epochs)
                end_time_train = start_time_train + ((N_epochs + 1) / (epoch + 1)) * (time.time() - start_time_train)
                end_time_train = datetime.datetime.fromtimestamp(int(end_time_train)).strftime(' %Y-%m-%d %H:%M:%S')
                print('    Step', epoch, ': Loss_train =', format(loss_train, '.4E'), ': Loss_test =', format(loss_test, '.4E'), " -  Estimated end:", end_time_train)

        print("Loss_train (final)=", format(best_loss_train, '.4E'))
        print("Loss_test (final)=", format(best_loss_test, '.4E'))

        print("Computation time for training (h:min:s):", str(datetime.timedelta(seconds=int(time.time() - start_time_train))))

        return (Loss_train, Loss_test, best_model)

class Integrate(Train, NA):

    def integrate(self, model, name, save_fig):
        """Prints the values of the Loss along the epochs, trajectories and errors.
        Inputs:
        - Ltr: List containing the values of Loss_train along the epochs
        - Lte: List containing the values of Loss_test along the epochs
        - model: Best model learned during training, Loss_train and Loss_test
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        def write_size():
            """Changes the size of writings on all windows"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            plt.legend(fontsize=7)
            pass

        def write_size3D():
            """Changes the size of writings on all windows - 3d variant"""
            axes = plt.gca()
            axes.title.set_size(7)
            axes.xaxis.label.set_size(7)
            axes.yaxis.label.set_size(7)
            axes.zaxis.label.set_size(7)
            plt.xticks(fontsize=7)
            plt.yticks(fontsize=7)
            axes.zaxis.set_tick_params(labelsize=7)
            plt.legend(fontsize=7)
            pass

        start_time_integrate = time.time()

        Model , Loss_train , Loss_test = model[0] , model[1] , model[2]

        print(" ")
        print(100 * "-")
        print("Integration...")
        print(100 * "-")

        fig = plt.figure()

        ax = fig.add_subplot(2, 1, 2)
        plt.plot(range(len(Loss_train)), Loss_train, color='green', label='$Loss_{train}$')
        plt.plot(range(len(Loss_test)), Loss_test, color='red', label='$Loss_{test}$')
        plt.grid()
        plt.legend()
        plt.yscale('log')
        plt.title('Evolution of the Loss function (MLP)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        write_size()

        def Flow_hat(t , y):
            """Vector field learned with the neural network
            Inputs:
            - t: Float - Time
            - y: Array of shape (d,) - Space variable"""
            #y = torch.tensor(y).reshape(d, 1)
            #y.requires_grad = True
            t_tensor = torch.tensor([[t]]).float()
            h_tensor = torch.tensor([[h_simul]]).float()
            eps_tensor = torch.tensor([[eps_simul]]).float()
            y = torch.tensor(y).float().reshape(d,1)
            z = Model(t_tensor , y)
            z = z.detach().numpy()
            z = np.array(z, dtype=np.float64)
            return z.reshape(d, )

        TT = np.arange(0, T_simul+h_simul, h_simul)

        # Integration with DOP853 (good approximation of exact flow)
        start_time_exact = time.time()
        Y_exact = NA.ODE_Solve(ti = 0 ,  tf = T_simul , h = h_simul , epsilon = eps_simul , Yinit = y0start(dyn_syst))
        print("Integration time of ODE with DOP853 (one trajectory - h:min:s):", datetime.timedelta(seconds=time.time() - start_time_exact))

        # Integration with learned flow
        start_time_app = time.time()
        Y_app = np.zeros_like(Y_exact)
        Y_app[:,0] = y0start(dyn_syst)
        for n in range(np.size(TT)-1):
            Y_app[:,n+1] = Flow_hat(TT[n+1] , y0start(dyn_syst))
            #Y_app[:,n+1] = Flow_hat(TT[n+1])

        print("Integration time of ODE with learned flow (h:min:s):", str(datetime.timedelta(seconds=time.time() - start_time_app)))

        print("   ")
        # Error computation between trajectory ploted with f for DOP853 and F_app for numerical method at stroboscopic times
        norm_exact = np.linalg.norm(np.array([np.linalg.norm((Y_exact)[:, i]) for i in range((Y_exact).shape[1])]) , np.infty) # Norm of the exact solution
        err_app = np.array([np.linalg.norm((Y_exact - Y_app)[:, i]) for i in range((Y_exact - Y_app).shape[1])])
        Err_app = np.linalg.norm(err_app, np.infty)/norm_exact
        print("Relative error between trajectories ploted with exact flow and learned flow:", format(Err_app, '.4E'))

        if d == 1:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            Y_exact = Y_exact.reshape(np.size(Y_exact),)
            Y_app = Y_app.reshape(np.size(Y_app), )
            plt.plot(TT, Y_exact, color='black', linestyle='dashed', label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(TT, Y_app, color="green", label=r"$\varphi_{t,\theta}^{f}(y_0)$")
            plt.ylim(np.min([np.min(Y_app), np.min(Y_exact)]),np.max([np.max(Y_app), np.max(Y_exact)]))
            plt.xlabel("$t$")
            plt.ylabel("$y$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]), max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$" + r"$\varphi_{t,\theta}^{f}(y_0) - \varphi_{t}^{f}(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.7, w * 1.7)

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

        if d == 2:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[0, :], Y_exact[1, :], color='black', linestyle='dashed',label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(Y_app[0, :], Y_app[1, :], color='green', label=r"$\varphi_{t,\theta}^{f}(y_0)$")
            plt.xlabel("$y_1$")
            plt.ylabel("$y_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]),max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$"+ r"$\varphi_{t,\theta}^{f}(y_0) - \varphi_{t}^{f}(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.7, w * 1.7)

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            if dyn_syst == "VDP":
                Y_exact_VC, Y_app_VC = np.zeros_like(Y_exact), np.zeros_like(Y_app)
                for n in range(np.size(TT)):
                    VC = np.array([[np.cos(TT[n] / eps_simul), np.sin(TT[n] / eps_simul)], [-np.sin(TT[n] / eps_simul), np.cos(TT[n] / eps_simul)]])
                    Y_exact_VC[:, n], Y_app_VC[:, n] = VC @ Y_exact[:, n], VC @ Y_app[:, n]

                plt.figure()
                plt.plot(np.squeeze(Y_exact_VC[0, :]), np.squeeze(Y_exact_VC[1, :]), label="Exact solution",  color="black", linestyle="dashed")
                plt.plot(np.squeeze(Y_app_VC[0, :]), np.squeeze(Y_app_VC[1, :]), label="Learned flow", color="green")
                plt.grid()
                plt.legend(loc = 'upper right')
                plt.xlabel("$q$")
                plt.ylabel("$p$")
                plt.title("$\epsilon = $" + str(eps_simul))
                plt.axis("equal")
                if save_fig == True:
                    plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
                else:
                    plt.show()


        if d == 4:
            plt.subplot(2, 2, 1)
            plt.title("Trajectories")
            plt.axis('equal')
            plt.plot(Y_exact[1, :], Y_exact[3, :], color='black', linestyle='dashed',label=r"$\varphi_{t}^f(y_0)$")
            plt.plot(Y_app[1, :], Y_app[3, :], color='green', label=r"$\varphi_{\theta}^{f}(t,y_0)$")
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.legend()
            plt.grid()
            write_size()
            plt.subplot(2, 2, 2)
            plt.title("Comparison of local errors")
            plt.ylim(min(err_app[1:]),max(err_app[1:]))
            plt.yscale('log')
            plt.plot(TT, err_app, color="orange", label="$|$"+ r"$\varphi_{\theta}^{f}(t,y_0) - \varphi_{t}^{f}(y_0) |$")
            plt.xlabel("t")
            plt.ylabel("Local error")
            plt.legend()
            plt.grid()
            write_size()

            f = plt.gcf()
            dpi = f.get_dpi()
            h, w = f.get_size_inches()
            f.set_size_inches(h * 1.7, w * 1.7)

            if save_fig == True:
                plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            Y_exact_VC, Y_app_VC = np.zeros_like(Y_exact), np.zeros_like(Y_app)
            for n in range(np.size(TT)):
                # VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                VC = np.array([[np.cos(TT[n] / eps_simul), 0, np.sin(TT[n] / eps_simul), 0], [0, 1, 0, 0],
                               [-np.sin(TT[n] / eps_simul), 0, np.cos(TT[n] / eps_simul), 0], [0, 0, 0, 1]])
                Y_exact_VC[:, n], Y_app_VC[:, n]= VC @ Y_exact[:, n], VC @ Y_app[:, n]

            plt.figure()
            plt.plot(np.squeeze(Y_exact_VC[1, :]), np.squeeze(Y_exact_VC[3, :]), label="Exact solution", color="black", linestyle="dashed")
            plt.plot(np.squeeze(Y_app_VC[1, :]), np.squeeze(Y_app_VC[3, :]), label="Learned flow", color="green")
            plt.grid()
            plt.legend()
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.title("$\epsilon = $" + str(eps_simul))
            plt.axis("equal")
            if save_fig == True:
                plt.savefig(name + "_Variable_change" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()

            plt.figure()
            H_exact = (1/(2*eps_simul))*(Y_exact_VC[0,:]**2+Y_exact_VC[2,:]**2) + (1/2)*(Y_exact_VC[1,:]**2+Y_exact_VC[3,:]**2) + (Y_exact_VC[0,:]**2*Y_exact_VC[1,:] - Y_exact_VC[1,:]**3/2)
            H_app = (1/(2*eps_simul))*(Y_app_VC[0,:]**2+Y_app_VC[2,:]**2) + (1/2)*(Y_app_VC[1,:]**2+Y_app_VC[3,:]**2) + (Y_app_VC[0,:]**2*Y_app_VC[1,:] - Y_app_VC[1,:]**3/2)
            plt.plot(TT, np.abs((H_exact-H_exact)/H_exact), label="Exact solution", color="black",linestyle="dashed")
            plt.plot(TT, np.abs((H_app-H_exact)/H_exact), label="Learned solution", color="green")
            plt.grid()
            plt.legend()
            plt.xlabel("$t$")
            plt.ylabel("Hamiltonian error")
            plt.yscale('log')
            #plt.xlim((0, T_simul))
            plt.title("Hamiltonian error (relative) - $\epsilon = $" + str(eps_simul))
            if save_fig == True:
                plt.savefig(name + "_Hamiltonian_error" + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
            else:
                plt.show()



        #plt.show()


        print("Computation time for integration (h:min:s):",
              str(datetime.timedelta(seconds=int(time.time() - start_time_integrate))))

        pass

class Trajectories(Integrate):
    """Class for the study of convergence of trajectories"""

    def traj(self, model, name, save_fig):
        """Prints the global errors according to the step of the numerical method
        Inputs:
        - model: Model learned during training
        - name: Character string - Potential name of the graph
        - save_fig: Boolean - Saves or not the figure"""

        EEPS = np.exp(np.linspace(np.log(eps[0]), np.log(eps[1]), 16))
        ERR_avg = []  # Global errors between exact flow of the averaged system and exact flow of the system
        EEPS_avg = [] # Associated high oscillation parameters
        ERR_app = []  # Global errors between exact flow of the averaged system and numerical flow with F_app at stroboscopic times
        EEPS_app = [] # Associated high oscillation parameters

        Model = model[0]

        for eepsilon in EEPS:
            print(" epsilon = {} \r".format(format(eepsilon, '.4E')), end="")

            def Fhat(t, y):
                """Vector field learned with the neural network
                Inputs:
                - t: Float - Time
                - y: Array of shape (d,) - Space variable"""
                y = torch.tensor(y).reshape(d, 1)
                y.requires_grad = True
                eps_tensor = torch.tensor([[np.float(eepsilon)]])
                z = Model(y, eps_tensor)
                z = z.detach().numpy()
                z = np.array(z, dtype=np.float64)
                return z.reshape(d, )

            # Integration with DOP853 (approximation of the exact flow)

            #Y_exact = NA.ODE_Solve(T = T_simul + 2 * np.pi * eepsilon , epsilon = eepsilon , Yinit = y0start(dyn_syst) , h = 2*np.pi*eepsilon)
            if num_meth == "Forward Euler":
                Y_exact_avg = NA.ODE_Solve_Av(T = T_simul + 2 * np.pi * eepsilon, epsilon = eepsilon, Yinit = y0start(dyn_syst), h = 2 * np.pi * eepsilon)
            if num_meth == "MidPoint":
                Y_exact_avg = NA.ODE_Solve_Av_1(T = T_simul + 2 * np.pi * eepsilon, epsilon=eepsilon, Yinit = y0start(dyn_syst), h=2 * np.pi * eepsilon)

            Y_exact_strob = NA.ODE_Solve(T = T_simul + 2 * np.pi * eepsilon , epsilon = eepsilon , Yinit = y0start(dyn_syst) , h = 2*np.pi*eepsilon)
            Y_app_strob = NA.ODE_Solve_Num(T = T_simul , epsilon = eepsilon, Yinit = y0start(dyn_syst), meth = num_meth, F = Fhat)

            # Computation of the norms of the exact solutions for the computation of relative errors
            norm_avg_strob = np.linalg.norm(np.array([np.linalg.norm((Y_exact_avg)[:, i]) for i in range((Y_exact_avg).shape[1])]),np.infty)  # Norm of the exact solution
            norm_sol_strob = np.linalg.norm(np.array([np.linalg.norm((Y_exact_strob)[:, i]) for i in range((Y_exact_strob).shape[1])]),np.infty)  # Norm of the exact solution at stroboscopic times

            # Computation of the error between exact flow of the averaged system and exact flow of the system
            err_avg = np.array([np.linalg.norm((Y_exact_strob - Y_exact_avg)[:, i]) for i in range((Y_exact_strob - Y_exact_avg).shape[1])])
            Err_avg = np.linalg.norm(err_avg, np.infty) / norm_sol_strob
            if Err_avg < 1:
                ERR_avg = ERR_avg + [Err_avg]
                EEPS_avg = EEPS_avg + [eepsilon]

            # Computation of the error between exact flow of the averaged system and numerical flow with F_app at stroboscopic times
            err_app = np.array([np.linalg.norm((Y_app_strob - Y_exact_strob)[:, i]) for i in range((Y_exact_strob - Y_app_strob).shape[1])])
            Err_app = np.linalg.norm(err_app, np.infty) / norm_sol_strob
            if Err_app < 1:
                ERR_app = ERR_app + [Err_app]
                EEPS_app = EEPS_app + [eepsilon]


        plt.figure()
        plt.title("Error between trajectories with " + num_meth)
        if len(ERR_avg) > 0:

            plt.scatter(EEPS_avg, ERR_app, label=r"$Max_{0 \leqslant n \leqslant N} \left| \left(\Phi_{2\pi\epsilon}^{F_{app}}\right)^n(y_0) - \varphi_{2\pi n\epsilon}^f(y_0) \right|$", marker="s", color="green")
        if len(ERR_app) > 0:
            if num_meth == "Forward Euler":
                plt.scatter(EEPS_avg, ERR_avg,label=r"$Max_{0 \leqslant n \leqslant N} \left| \varphi_{2\pi n\epsilon}^{F^{[0]}}(y_0) - \varphi_{2\pi n\epsilon}^f(y_0) \right|$", marker="s", color="red")
            if num_meth == "MidPoint":
                plt.scatter(EEPS_avg, ERR_avg, label=r"$Max_{0 \leqslant n \leqslant N} \left| \varphi_{2\pi n\epsilon}^{F^{[1]}}(y_0) - \varphi_{2\pi n\epsilon}^f(y_0) \right|$", marker="s", color="red")
        plt.legend()
        plt.grid()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("High oscillation parameter $\epsilon$")
        plt.ylabel("Global error")
        if save_fig == True:
            plt.savefig(name + dtime.now().strftime(' - %Y-%m-%d %H:%M:%S') + '.pdf', dpi=(200))
        else:
            plt.show()
        pass




def ExData(name_data="DataODE_FLow_Averaging"):
    """Creates data y0, y1 with the function solvefEDOData
    with the chosen vector field at the beginning of the program
    Input:
    - name_data: Character string - Name of the registered tuple containing the data (default: "DataEDO")"""
    DataODE = DataCreate.Data(K_data)
    torch.save(DataODE, name_data)
    pass

def ExTrain(name_model='model_Flow_Averaging_PINN'):
    """Launches training and computes Loss_train, loss_test and best_model with the function Train().train
    Saves the files Loss_train, Loss_test and best_model with a given name
    Inputs (character strings):
    - name_model: Name of the file saved for best_model (default: "best_model")
    - name_data: Name of the file containing the created data (default: "DataEDO") used for training"""
    Loss_train, Loss_test, best_model = Train().train(model=NN())
    torch.save((best_model,Loss_train,Loss_test), name_model)
    pass

def ExIntegrate(name_model="model_Flow_Averaging_PINN", name_graph="Simulation_Flow_Averaging_PINN", save=False):
    """Launches integration of the main equation and modified equation with the chosen model
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app, and Loss_train/Loss_test
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Integrate().integrate(model=Lmodel, name=name_graph,save_fig=save)
    pass

def ExTraj(name_model="model", name_graph="Simulation_Convergence_Trajectories", save=False):
    """plots the curves of convergence between the trajectories integrated with f and F_app with the numerical method chosen
    Inputs:
    - name_model: Character string - Name of the model made with neural network chosen for integration with F_app
    - name_graph: Character string - Name of the graph which will be registered
    - save: Boolean - Saves the figure or not (default: False)"""
    Lmodel = torch.load(name_model)
    Trajectories().traj(model=Lmodel, name=name_graph, save_fig=save)
    pass


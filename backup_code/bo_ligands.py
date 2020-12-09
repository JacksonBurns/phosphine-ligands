"""
@uthor: Himaghna, 22nd November 2019
Description: Perform Bayesian Optimization on the California data set
"""

from argparse import ArgumentParser

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import torch
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


# Globals
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
N_STD_DEV = 100

# RC PArams
plt.rcParams['svg.fonttype'] = 'none'
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)


def get_gpr_model(X, y, model=None):
    """
    Fit a gpr model to the data or update the model to new data
    Params ::
    X: (sx1) Tensor: Covariates
    y: (sx1) Tensor: Observations
    model: PyTorch SingleTaskGP model: If model is passed, X and y are used to 
        update it. If None then model is trained on X and y. Default is None
    Return ::
    model: PyTorch SingleTaskGP model: Trained or updated model. 
        Returned in train mode
    mll: PyTorch MarginalLogLikelihood object: Returned in train mode
    """
    
    if model is None:
        # set up model
        model = SingleTaskGP(X, y)
    else:
        # update model with new observations
        model = model.condition_on_observations(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(X)
    # begin training
    model.train();
    mll.train();
    fit_gpytorch_model(mll);
    return model, mll

def plot_testing(model, X_train, y_train, X_test, target, x_dim):
    '''
    Test the surrogate model with model, test_X and new_X
    '''

    # Initialize plot
    font = {'size'   : 20}
    matplotlib.rc('font', **font)
    matplotlib.rcParams['axes.linewidth'] = 1.5
    matplotlib.rcParams['xtick.major.size'] = 8
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['ytick.major.size'] = 8
    matplotlib.rcParams['ytick.major.width'] = 2
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    # set up model in eval mode
    model.eval();

    with torch.no_grad():
        posterior = model.posterior(X_test)
        # Get upper and lower confidence bounds (2 std from the mean)
        lower, upper = posterior.mvn.confidence_region()
        ax.plot(X_test, posterior.mean.cpu().numpy(), \
            'b', label='Posterior Mean')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(X_test[:, x_dim], lower.cpu().numpy(), \
            upper.cpu().numpy(), alpha=0.5, label = '95% Credibility')
        
        # Plot training points as black stars
        ax.scatter(X_train[:, x_dim].cpu().numpy(), 
            y_train.cpu().numpy(),
            s=120, c= 'k', marker = '*', 
            label = 'Training Data')
        
    ax.set_xlabel(f'{target}')
    ax.set_ylabel('Selectivity')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()
    acq_func = ExpectedImprovement(model, best_f=y_train.max(), maximize=True)
    plot_acq_func(acq_func, X_train, x_dim=x_dim)

def plot_acq_func(acq_func, X_test, x_dim):
    # compute acquisition function values at test_X
    test_acq_val = acq_func(X_test.view((X_test.shape[0], 1, X_test.shape[1])))
    print(test_acq_val)
    # Initialize plot
    hist_fig, ax = plt.subplots(figsize=(12, 6))
    with torch.no_grad():
        ax.scatter(X_test[:, x_dim].cpu().numpy(), 
        test_acq_val.cpu().detach(), c='blue', s=1.2,
            alpha=0.7, label='Acquisition (EI)')
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-2,2) )
    ax.set_xlabel('x')
    ax.set_ylabel(r'$ \alpha$')    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def optimize_loop(model, loss, X_train, y_train, bounds, n_samples=10):
    best_value = y_train.max()
    acq_func = qExpectedImprovement(model, best_f= best_value)
    X_new, acq_value = optimize_acqf(acq_func, bounds=bounds, q=20,
        num_restarts=10, raw_samples=76)
    #X_new = X_new.view((n_samples,-1))
    print(X_new)
    return X_new
    
if __name__ == "__main__":
    # import the data
    parser = ArgumentParser()
    parser.add_argument('-x', help='Path of X.p')
    parser.add_argument('-y', help='Path of y.p')
    parser.add_argument('-dn', '--descriptor_names', 
        help='Path of pickle with descriptor names')
    parser.add_argument('-np', '--n_points', 
       type=int,
       help='Number of pooints to sample along each descriptor') 
    args = parser.parse_args()
    
    X = torch.from_numpy(pickle.load(open(args.x, "rb"))).type(dtype)
    y = torch.from_numpy(pickle.load(open(args.y, "rb"))).type(dtype).reshape(-1, 1)
    descriptor_names = pickle.load(open(args.descriptor_names, "rb"))

    #explore_dataset(processed_data) # explore dataset
    # normalize X and y
    y_scale = y.std(dim=0)
    y_mean = y.mean(dim=0)
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0)
    X = (X - X_mean) / X_std
    #y = (y - y_mean) / y_scale

    # set up GPR model
    gpr_model, gpr_mll = get_gpr_model(X=X, y=y)
            
    opt_bounds = torch.cat((torch.min(X, dim=0).values.view(1, -1), 
                            torch.max(X, dim=0).values.view(1, -1)), 
                           dim=0)    # need to be (2 x D) tensor
    X_new = optimize_loop(model=gpr_model, 
                  loss=gpr_mll, 
                  X_train=X, 
                  y_train=y, 
                  bounds=opt_bounds, 
                  n_samples=10)
    X_new = (X_new * X_std + X_mean).numpy()

    # for descr_id, descr_name in enumerate(descriptor_names):
    #     plt.hist(X_new[:, descr_id], color='red')
    #     plt.ylabel('Frequency', fontsize=28)
    #     plt.xlabel(descr_name, fontsize=28)
    #     plt.xticks(fontsize=20)
    #     plt.yticks(fontsize=20)
    #     plt.show()
    
    
    
    
    # X_numpy = X.numpy()  # for calculating median using numpy
    # X_median = list(np.median(X_numpy, axis=0))
    # for descr_id, descriptor in enumerate(descriptor_names):
    #     median_descriptor = X_median[descr_id]
    #     grid = np.linspace(-N_STD_DEV * median_descriptor, 
    #                         N_STD_DEV * median_descriptor, 
    #                         args.n_points)
    #     new_X = []
    #     for grid_id, grid_val in enumerate(grid):
    #         new_sample = X_median.copy()
    #         new_sample[descr_id] = grid_val
    #         new_X.append(new_sample)
    #     new_X = torch.FloatTensor(new_X).type(dtype)                           
        
    #     plot_testing(
    #         gpr_model,
    #         X_train=X,
    #         y_train=y,            
    #         X_test=new_X, 
    #         target=descriptor,
    #         x_dim=descr_id)
        
    

    # do some optimization!
    
    N_OPT_STEPS = 10
    opt_bounds = torch.stack([X.min(dim=0).values, X.max(dim=0).values])
    max_val, upper_confidence, lower_confidence = [], [], []
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.90)
    for _ in range(N_OPT_STEPS):
                # get the point which has the maximum posterior and the variance to it
        gpr_model.eval();
        posterior = gpr_model.posterior(X_train)
        lower, upper = posterior.mvn.confidence_region()
        max_posterior, index = posterior.mean.max(dim=0)
        max_val.append(float(max_posterior * y_scale + y_mean))
        upper_confidence.append(float(upper[index] * y_scale + y_mean))
        lower_confidence.append(float(lower[index] * y_scale + y_mean))
        gpr_model, gpr_mll, X_train, y_train, X_test, y_test = optimize_loop(
            model=gpr_model, \
            loss=gpr_mll, X_train=X_train, y_train=y_train, \
                #X_test=X_test, y_test=y_test, \
                    bounds=opt_bounds)

    plt.plot([_ for _ in range(N_OPT_STEPS)], max_val, \
        'go--', linewidth=2, markersize=12)
    plt.fill_between([_ for _ in range(N_OPT_STEPS)], lower_confidence, \
            upper_confidence, alpha=0.5, label = '95% Credibility')
    plt.show()

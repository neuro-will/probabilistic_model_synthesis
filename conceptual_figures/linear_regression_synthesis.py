# In these simulations, we desire to perform model synthesis on linear example systems.  In particular example
# system s produces its i^th output, y^s_i \in R, from its i^th input x^s_i \in R^5 according to:
#
#   y_i = beta^j' * x_i + \ep_i,
#
#   where x_i is input (pulled from a 5-d standard Normal distribution) and ep_i is iid, pulled
#   from a 1-d standard Normal) and beta^j is a vector of regression weights pulled from a N(mu, cov) distribution
#   where mu is a 5-d vector and cov is a diagonal 5x5 matrix. This N(mu, cov) distribution represents the ``ground
#   truth'' CPD.
#
# From each example system, we generate *ONE AND ONLY ONE* sample.  Using these data we want to learn the CPD by
# applying DPMS, and we seek to examine performance as we vary the number of example systems.
#
# A couple of technical notes:
#
#   1) We assume the only unknowns for each example system are beta (we assume the variance of the observation
#      noise is known to be 1).
#
#   2) In this simple example, we assume there are no properties (or equivalently all example systems have the same
#      properties).  When discrete properties are introduced, the findings here should generalize, as we would
#      essentially be performing the synthesis shown here for each value of conditioning properties.
#

import copy
import itertools
import multiprocessing
from multiprocessing import Pool
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from janelia_core.ml.torch_distributions import CondGaussianDistribution
from janelia_core.ml.extra_torch_modules import ConstantBoundedFcn
from janelia_core.ml.extra_torch_modules import ConstantRealFcn

from probabilistic_model_synthesis.distributions import SampleLatentsGaussianVariationalPosterior

# ==================================================================================================================
# High level parameters here (see where the main script starts below for all other parameters)
# ==================================================================================================================

LOAD_PREV_RESULTS = True
SAVE_FILE = r'/Volumes/bishoplab/projects/probabilistic_model_synthesis/results/simulation/reg_with_varying_no_of_example_systems/reg_synthesis_with_varying_n_ex_systems.pkl'
FIG_SAVE_FILE = r'/Users/bishopw/Desktop/reg_synthesis_with_varying_n_ex_systems.eps'


# ==================================================================================================================
# Helper functions
# ==================================================================================================================


def compute_metrics(params, rs):

    cpd = rs['cpd']

    cpd_mn = cpd(torch.ones(1)).detach().squeeze().numpy()
    cpd_std = cpd.std_f(torch.ones(1)).detach().squeeze().numpy()

    true_mn = params['true_beta_mn'].numpy()
    true_std = params['true_beta_std'].numpy()

    mn_rmse = np.sqrt(np.mean((true_mn - cpd_mn)**2))
    std_rmse = np.sqrt(np.mean((true_std - cpd_std)**2))
    mean_cpd_std = np.mean(cpd_std)
    geomean_cpd_std = np.prod(cpd_std)**(1/len(cpd_std))

    return {'mn_rmse': mn_rmse, 'std_rmse': std_rmse,
            'mean_cpd_std': mean_cpd_std, 'geomean_cpd_std': geomean_cpd_std}


def run_simulation(params):

    # Set random seed - this is important for reproducability and for ensuring there are differences between simulations
    torch.manual_seed(params['rnd_seed'])

    # ==================================================================================================================
    # Generate data

    # Generate regression weights for each example system
    x_dim = len(params['true_beta_mn'])
    true_ex_system_weights = params['true_beta_mn'] + params['true_beta_std']*torch.randn(params['n_ex_systems'], x_dim)

    #  Generate x and y data for each example system
    data = [None] * params['n_ex_systems']
    for s_i, beta_i in enumerate(true_ex_system_weights):
        x_i = torch.randn(params['n_smps_per_system'], x_dim)
        expected_output = torch.sum(x_i * beta_i, dim=1)
        y_i = expected_output + params['obs_std'] * torch.randn(params['n_smps_per_system'])
        data[s_i] = (x_i, y_i)

    # ==================================================================================================================
    # Setup everything for synthesis

    # Initialize CPD - the mean and standard deviation functions are just constant
    # since we have no properties to condition on
    cpd = CondGaussianDistribution(mn_f=ConstantRealFcn(np.ones([x_dim])),
                                   std_f=ConstantBoundedFcn(.01 * np.ones([x_dim]),
                                                            10 * np.ones([x_dim]),
                                                            3 * np.ones([x_dim])))

    # Initialize approximate posteriors for each example system
    approx_posts = [SampleLatentsGaussianVariationalPosterior(x_dim, 1) for _ in range(params['n_ex_systems'])]

    # ==================================================================================================================
    # Perform synthesis
    syn_params = list(cpd.parameters()) + list(itertools.chain(*[list(post.parameters()) for post in approx_posts]))

    # Setup optimizer
    optimizer = torch.optim.Adam(params=syn_params, lr=params['init_lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=params['milestones'], gamma=params['gamma'])
    # Constant for calculating log-likelihoods
    constant = -.5 * params['n_smps_per_system'] * np.log(2 * np.pi * (params['obs_std'] ** 2))

    for i in range(params['n_train_its']):

        optimizer.zero_grad()

        # Sample from approximate posteriors over regression weights for each example system
        post_mn_smps = [post.sample([0]) for post in approx_posts]

        # Calculate log-likelihood of observed data conditioned on sampled weights for each example system
        elbo = 0
        for data_i, mn_i in zip(data, post_mn_smps):
            x_i, y_i = data_i
            pred_mn_i = torch.sum(x_i * mn_i, dim=1)
            ll_i = constant - .5 * torch.sum(((y_i - pred_mn_i) / params['obs_std']) ** 2)
            elbo += ll_i

        # Calculate KL divergence between each approximate posterior and cpd
        cpd_mn = cpd(torch.zeros(1)).squeeze()
        cpd_var = cpd.std_f(torch.zeros(1)).squeeze() ** 2
        for post in approx_posts:
            elbo -= post.kl_btw_diagonal_normal([0], cpd_mn, cpd_var)

        neg_elbo = -1 * elbo
        neg_elbo.backward()
        optimizer.step()

        #if i % 100 == 0:
        #    print('It: ' + str(i) + ', ELBO: ' + str(elbo.item()))

        scheduler.step()

    # ==================================================================================================================
    # Measure how well we learned the cpd

    rs = {'cpd': cpd,
          'approx_posts': approx_posts}

    metrics = compute_metrics(params, rs)

    print('Finished simulation for ' + str(params['n_ex_systems']) + ' example systems.')

    # ==================================================================================================================
    # Return results

    return rs, metrics



# ==================================================================================================================
# Main script starts here - see below for setting parameters
# ==================================================================================================================

if __name__ == '__main__':

    if not LOAD_PREV_RESULTS:

        # ==================================================================================================================
        # Parameters go here
        # ==================================================================================================================
        ps = dict()

        # Mean and standard deviation of the true prior over regression weights.  (These
        # parameters implicitly define the dimensionality of the x-data).

        ps['true_beta_mn'] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        ps['true_beta_std'] = 1 * torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

        # Standard deviation of noise for observations
        ps['obs_std'] = 1

        # Number of samples we observe from each example system
        ps['n_smps_per_system'] = 1

        # Number of example systems we try fitting to in each simulation - should be a list of values we want to try
        ps['n_ex_systems'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50] #, 20, 50, 100]

        # Number of times we repeat simulations at for a fixed number of example systems
        ps['n_repeats'] = 30

        # Parameters for optimization
        ps['init_lr'] = .1
        ps['milestones'] = [500]
        ps['gamma'] = .1
        ps['n_train_its'] = 1500

        # ==============================================================================================================
        # Run simulations
        # ==============================================================================================================

        # Generate copy of parameters for each simulation we want to run in parallel
        par_params = []
        rnd_seed_cnt = 0
        for n_ex_systems in ps['n_ex_systems']:
            for r_i in range(ps['n_repeats']):
                params_i = copy.deepcopy(ps)
                params_i['n_ex_systems'] = n_ex_systems
                params_i['rnd_seed'] = rnd_seed_cnt
                par_params.append(params_i)

                rnd_seed_cnt += 1

        # Run simulations
        n_cores = multiprocessing.cpu_count()
        print('Found ' + str(n_cores) + ' cores.')
        print('Running ' + str(len(par_params)) + ' simulations total.')

        #sim_rs = [run_simulation(p) for p in par_params]
        with Pool(n_cores) as pool:
            sim_rs = pool.map(run_simulation, par_params)

        # Organize results

        mn_rmse = np.zeros([ps['n_repeats'], len(ps['n_ex_systems'])])
        std_rmse = np.zeros([ps['n_repeats'], len(ps['n_ex_systems'])])
        mean_std = np.zeros([ps['n_repeats'], len(ps['n_ex_systems'])])
        geomean_std = np.zeros([ps['n_repeats'], len(ps['n_ex_systems'])])

        for e_i, n_ex_systems in enumerate(ps['n_ex_systems']):
            rs_inds = np.argwhere([True if params['n_ex_systems'] == n_ex_systems else False
                                   for params in par_params]).squeeze()
            print(rs_inds)
            for r_i, ind in enumerate(rs_inds):
                mn_rmse[r_i, e_i] = sim_rs[ind][1]['mn_rmse']
                std_rmse[r_i, e_i] = sim_rs[ind][1]['std_rmse']
                mean_std[r_i, e_i] = sim_rs[ind][1]['mean_cpd_std']
                geomean_std[r_i, e_i] = sim_rs[ind][1]['geomean_cpd_std']

        # Save results
        with open(SAVE_FILE, 'wb') as f:
            full_rs = dict()
            full_rs['ps'] = ps
            full_rs['sim_rs'] = sim_rs
            full_rs['mn_rmse'] = mn_rmse
            full_rs['std_rmse'] = std_rmse
            full_rs['mean_std'] = mean_std
            full_rs['geomean_std'] = geomean_std
            pickle.dump(full_rs, f)
    else:
        print('Loading previous simulation results from ' + SAVE_FILE)
        with open(SAVE_FILE, 'rb') as f:
            saved_rs = pickle.load(f)
            sim_rs = saved_rs['sim_rs']
            mn_rmse = saved_rs['mn_rmse']
            std_rmse = saved_rs['std_rmse']
            mean_std = saved_rs['mean_std']
            geomean_std = saved_rs['geomean_std']
            ps = saved_rs['ps']

    # ==================================================================================================================
    # Generate figure
    # ==================================================================================================================


    avg_mn_rmse = np.mean(mn_rmse, axis=0)
    stderr_mn_rmse = np.std(mn_rmse, axis=0)/np.sqrt(ps['n_repeats'])
    avg_std_rmse = np.mean(std_rmse, axis=0)
    stderr_std_rmse = np.std(std_rmse, axis=0) / np.sqrt(ps['n_repeats'])

    avg_geomean_std = np.mean(geomean_std, axis=0)
    stderr_geomean_std = np.std(geomean_std, axis=0) / np.sqrt(ps['n_repeats'])

    fig = plt.figure(figsize=[10, 4])
    ax = plt.subplot(1,2,1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.errorbar(ps['n_ex_systems'], avg_mn_rmse, yerr=stderr_mn_rmse, color='k')
    plt.ylim(0, 2.5)
    plt.xlabel('number of example systems')
    plt.ylabel('RMSE')
    plt.legend(['RMSE'])
    plt.title('recovery of mean')

    ax = plt.subplot(1, 2, 2)
    ax.errorbar(ps['n_ex_systems'], avg_std_rmse, yerr=stderr_std_rmse, color='k')
    ax.errorbar(ps['n_ex_systems'], avg_geomean_std, yerr=stderr_geomean_std, color='b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(0, 1.5)
    plt.xlabel('number of example systems')
    plt.ylabel('geometric mean / RMSE')
    plt.title('recovery of variances')
    plt.legend(['RMSE', 'geometric mean'])
    plt.show()

    fig.savefig(FIG_SAVE_FILE, format='eps')




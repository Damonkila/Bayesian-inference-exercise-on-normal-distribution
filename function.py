import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import scipy.special as ss
import sys
from IPython.display import display, Math

   


def log_prior(theta, threshold):

    mu , sd = theta
    
    if -50.0 < mu < 50.0 and 0.01 < sd < 10.0:
        return 0.0
    
    return -np.inf

## likelihood of normal distribution

def log_selection_bias(theta, threshold):
    mu, sd = theta
 

    z = (threshold - mu) / sd / np.sqrt(2)
    integral = (1.0 - ss.erf(z) )/ 2
    ## extreme value of log function
    if integral < 1e-200:
        return -np.inf 

    return np.log(integral)

def log_lh(theta, x):
    
    mu, sd = theta
    N =  np.size(x) # number of observations
    return - N / 2.0 * np.log(2.0 * np.pi) - N * np.log(sd) - np.sum( (x-mu)**2 ) / 2.0 / sd**2
    
def pdf(x,mu,sd):
    
    index = -0.5 * ( (x-mu) / sd )**2
    
    #norm = 1/ np.sqrt(2*np.pi) / sigma 
    #return norm * np.exp(index)  
    return np.exp(index)

def log_lhWithError(theta, x):

    #print(x.shape)
    
    mu, sd = theta
    norm = np.sqrt(2*np.pi) * sd
    N =  np.size(x[:,0]) # number of obs
    Ns = np.size(x[0,:]) # number of samples for single obs

    p_obs = np.sum( pdf(x, mu, sd), axis = 1)             ### probability of a single obs 
    #print(p_obs.shape)

    if p_obs.all() <=0:
        return -np.inf
 
    return - N * ( np.log(Ns * norm)) + np.sum( np.log(p_obs) )
    

def log_probability(theta, x, threshold, Error = False, Selec_Bias = False):
    lp = log_prior(theta, threshold)


    if not np.isfinite(lp):
        return -1 * np.inf
    
    if Selec_Bias:
        ls = -np.size(x) * log_selection_bias(theta, threshold)
    
        if not np.isfinite(ls):
            return -1 * np.inf
    else:
        ls = 0

    if Error:
        llh = log_lhWithError(theta, x)
    else :
        llh = log_lh(theta, x)
    
    if not np.isfinite(llh):
        return -np.inf
    
    return lp + ls + llh 

## extend data x to include measurement error samples 
def x_samples_assemble(x, error_sd, Nsample_for_each_obs,trial_index,N,path):

    N = np.size(x) 

    x_extended = np.zeros((N, Nsample_for_each_obs))

    for i in range(N):
        x_extended[i,:] = np.random.normal(x[i],error_sd, Nsample_for_each_obs)


    path_hist_error  = path + '/sd_Nsample=' + str(error_sd) + '_' + str(Nsample_for_each_obs)
            
    if not os.path.exists(path_hist_error):
        os.makedirs(path_hist_error)

    figpath = path_hist_error +'/' +'trial'+str(trial_index)+ '_'+str(N)  + 'hist_witherror_after_mean.png'
    plot_hist(N, x_extended.flatten(), figpath)


    return x_extended

def init_position(x,nwalkers,ndim):
    #est_mu = np.sum(x) / np.size(x)
    #est_sd = np.sum( (x - est_mu)**2  ) / np.size(x)
    est_mu = 1 
    est_sd = 0.2
    pos =  np.zeros((nwalkers, ndim))   
    pos[:,0] = np.random.rand(nwalkers)  + est_mu
    pos[:,1] = np.random.rand(nwalkers)  + est_sd

    return pos 

def steps(ndim, trial_index, N, steps, samples,path):
            
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    labels = ["mean", "sd"]
    for j in range(ndim):
        # the loop is the same size with the array but is [0, size -1 ], use for index
        ax = axes[j]
        ## alpha = scale factor
        ax.plot(samples[:, :, j], "k", alpha=0.3)
        #ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[j])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    figpath = path + '/' +'trial' + str(trial_index) +'_'+ str(N) + '_' + str(steps) + 'step.png'
    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()
    #plt.clf()
    #plt.cla()
def contour_plot(flat_samples,mu_true, sd_true, N, steps, trial_index, tau,labels, path):
    ### Contour plot
    
    fig = corner.corner(
        flat_samples, labels=labels, truths=[mu_true, sd_true]
    )


    figpath = path + '/' + 'trial'+str(trial_index)+'_'+ str(N) + '_' + str(steps)+'_'+ str(tau) + 'corr.png'
    
    plt.tight_layout()
    
    plt.savefig(figpath)
    plt.show()
    
    #plt.clf()
    #plt.cla()



def plot_hist(N, x, path):
   
    plt.title(str(N)+' observations histogram')
    plt.ylabel('counts')
    plt.xlabel('x')
    histogram = plt.hist(x,bins=100)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    #plt.clf()
    #plt.cla()

    
def plot_bias(N, x, path):
    plt.title(str(len(x))+'/'+str(N)+' observations histogram')
    plt.ylabel('counts')
    plt.xlabel('x')
    plt.xlim(-np.max(x),np.max(x))
    histogram = plt.hist(x,bins=100)
    #figname = path2 +'/' +'trial'+str(trial_index)+ '_'+str(N)  + 'hist_normal.png'
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def createPath(N, mu_true, sd_true, error_sd, Nsample_for_each_obs, threshold,Error = False, Selec_Bias= False):
    ########### Creating path and folders
    home = './mu_sd=' +str(mu_true) + '_' + str(sd_true) + '/'
    
    err_path = 'errSD_Nsample=' + str(error_sd) + '_' + str(Nsample_for_each_obs) + '/'
    bias_path = 'th=' + str(threshold) + '/'
    

    if Error and Selec_Bias:
        home = home + 'hybird/' + err_path + bias_path + str(N)

    elif Error:
        home = home + 'err/' + err_path + str(N)

    elif Selec_Bias:
        home = home + 'bias/' +  bias_path + str(N)
    else:
        home = home + 'none/' + str(N)

    path_folder = ['/hist', '/steps', '/corr']
    path = [home + string for string in path_folder] 

    for path_x in path:
        if not os.path.exists(path_x):
            os.makedirs(path_x)
    return path,home

def outputPara(home, N, mu_true,sd_true, error_sd, Nsample_for_each_obs, threshold, nwalkers, burnt_step, trial, Error = False, Selec_Bias= False, wseed = False):
    ###Output the parameters in the program 
    
    f = open(home + '/para.txt', 'w')
    f.write('Population =' + str(N))
    f.write('\nMean, SD =' + str(mu_true) + ', '+  str(sd_true) )
    if Error:
        f.write('\nError on each data: Yes ')
        f.write('\nError SD for each data =' + str(error_sd))
        f.write('\nNumber of samples for each data point =' + str(Nsample_for_each_obs))
    else:
        f.write('\nError on each data: None ')
    if Selec_Bias:

        f.write('\nBias on obs data: Yes ')
        f.write('\nThreshold for the obs data =' + str(threshold))

    else:
        f.write('\nBias on obs data: None ')
    
    f.write('\nNumber of walkers in para space =' + str(nwalkers))
    f.write('\nNumber of burnt steps =' + str(burnt_step))
    
    f.write('\nNumber of trial for the run =' + str(trial))
    
    if wseed:
        np.random.seed(7)
        f.write('\nSeed of Rand Num =' + str(7))
    else:
        f.write('\nSeed of Rand Num =None')

    f.close()

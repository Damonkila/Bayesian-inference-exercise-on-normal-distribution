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
    
    if -50.0 < mu < 50.0 and 0.1 < sd < 10.0:
        return 0.0
    
    #return 0.0
    return -np.inf

## likelihood of normal distribution

def log_selection_bias(theta, threshold):
    mu, sd = theta
 

    z = (threshold - mu) / sd / np.sqrt(2)
    integral = (1.0 - ss.erf(z) )/ 2
    ## extreme value of log function
    if integral <= 0:
        return -np.inf 

    return np.log(integral)

def log_lh(theta, x):
    
    mu, sd = theta
    N =  np.size(x) # number of observations
    #print(sd)
    return - N / 2.0 * np.log(2.0 * np.pi) - N * np.log(sd) - np.sum( (x-mu)**2 ) / 2.0 / sd**2
    
def pdf(x,mu,sd):
    
    index = -0.5 * ( (x-mu) / sd )**2
    
    #norm = 1/ np.sqrt(2*np.pi) / sigma 
    #return norm * np.exp(index)  
    return np.exp(index)

def log_lhWithError(theta, x):

    #print(x.shape)
    
    mu, sd = theta
    norm = 1.0 / (np.sqrt(2*np.pi) * sd)
    N =  np.size(x, axis = 0) # number of obs
    Ns = np.size(x, axis = 1) # number of samples for single obs

    p_obs = norm * np.sum( pdf(x, mu, sd), axis = 1)   / Ns           ### probability of a single obs 
    #print(p_obs.shape)

    if p_obs.all() <=0:
        return -np.inf
 
    return np.sum( np.log(p_obs) )
    

def log_probability(theta,N, x, threshold, Error = False, Selec_Bias = False):
    lp = log_prior(theta, threshold)


    if not np.isfinite(lp):
        return -1 * np.inf
    
    if Selec_Bias:
        ls = -N  * log_selection_bias(theta, threshold)
        #print('ls=',ls)
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
def x_samples_assemble(x, error_sd, Nsample_for_each_obs,trial_index, path):

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
    N = np.size( x )
    est_mu = np.sum(x) / N
    est_sd = np.sum( (x - est_mu)**2  ) / N
    #est_mu = 1 
    #est_sd = 0.2
    pos =  np.zeros((nwalkers, ndim))   
    pos[:,0] = np.random.rand(nwalkers) + est_mu
    pos[:,1] = np.random.rand(nwalkers)  + est_sd
    #print(pos[:,1])

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
    f.write('Population =%d' % (N))
    f.write('\nMean=%.1f, SD = %.1f,' % (mu_true, sd_true) )
    if Error:
        f.write('\nError on each data: Yes ')
        f.write('\nError SD for each data = %.1f' % error_sd)
        f.write('\nNumber of samples for each data point =%d' % Nsample_for_each_obs)
    else:
        f.write('\nError on each data: None ')
    if Selec_Bias:

        f.write('\nBias on obs data: Yes ')
        f.write('\nThreshold for the obs data =%.1f' % threshold)

    else:
        f.write('\nBias on obs data: None ')
    
    f.write('\nNumber of walkers in para space =%d' %nwalkers)
    f.write('\nNumber of burnt steps =%d' % burnt_step)
    
    f.write('\nNumber of trial for the run =%d' % trial)
    
    if wseed:
        #np.random.seed(7)
        f.write('\nSeed of Rand Num =%d' %7)
    else:
        f.write('\nSeed of Rand Num =None')

    f.close()

def pclor(mu_true, sd_true, error_sd, Nsample_for_each_obs, th, mu_low, mu_up, perr = False, psel = False):
    np.random.seed(7)
    threshold = mu_true + th * sd_true
    x = np.random.normal(mu_true,sd_true,500)
    #threshold = 0.3
    if psel:
        x = x[x > threshold]
    N = np.size(x)
    path = './a.png'
    if perr:
        trial_index = 1000
        x = x_samples_assemble(x, error_sd, Nsample_for_each_obs,trial_index, path)

    #theta = np.array([[0,1]]).reshape((2,1))
    
    mu = np.linspace(mu_low, mu_up,300)
    sd = np.linspace(0.1, 10,300)
    xx,yy = grid = np.meshgrid(mu,sd)

    a = np.zeros((300,300))

    for i in range(300):
        for j in range(300):

            a[i,j] = log_probability(np.array([xx[i,j],yy[i,j]]),N, x, threshold, Error = perr, Selec_Bias = psel)


    index = np. unravel_index(a.argmax(), a.shape)
    #print('hihihihh',index)
    print('max log prob = ',log_probability(np.array([xx[index],yy[index]]),N, x, threshold, Error = perr, Selec_Bias = psel))
    print('mu=',xx[index])
    print('sd=',yy[index])
    print('--------------------------------')
    theta = np.array([mu_true,sd_true])
    print('true para log prob = ',log_probability( theta, N, x, threshold, Error = perr, Selec_Bias = psel))
    print('mu=',mu_true)
    print('sd=',sd_true)
    plt.imshow(-1/a / np.max(-1/a),extent=[mu_low, mu_up,10,0],aspect='auto')
    plt.colorbar( label = 'Rescaled log probability')
    plt.xlabel('$\mu$')
    plt.ylabel('$\sigma$')
    plt.show()
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(xx, yy, -1/ a / np.max(-1/a))
    fig.colorbar(cp, label = 'Rescaled log probability')
    plt.xlabel('$\mu$')
    plt.ylabel('$\sigma$')
    plt.show()
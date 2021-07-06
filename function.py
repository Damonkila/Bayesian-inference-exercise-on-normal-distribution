import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import scipy.special as ss
from IPython.display import display, Math

def log_prior(theta, threshold):

    mu , sd = theta
    if -50.0 < mu < 50.0 and 0.1 < sd < 10.0:
        return 0.0
    
    return -np.inf


def log_SelectionBias(theta, threshold):
    mu, sd = theta
    z = (threshold - mu) / sd / np.sqrt(2)
    integral = (1.0 - ss.erf(z) )/ 2
    ## extreme value of log function
    if integral <= 0:
        return -np.inf 

    return np.log(integral)


def log_likelihood(theta, x):
    
    mu, sd = theta
    N =  np.size(x) # number of observations
    return - N / 2.0 * np.log(2.0 * np.pi) - N * np.log(sd) - np.sum( (x-mu)**2 ) / 2.0 / sd**2

## likelihood of normal distribution    
def pdf(x,mu,sd):
    norm = 1.0 / (np.sqrt(2*np.pi) * sd)
    
    return norm * np.exp( -0.5 * ( (x-mu) / sd )**2  )

def log_likelihoodWithError(theta, x):
    mu, sd = theta
    N =  np.size(x, axis = 0) # number of obs
    Ns = np.size(x, axis = 1) # number of samples for single obs
    p_obs = np.sum( pdf(x, mu, sd), axis = 1)   / Ns           ### probability of a single obs 

    if p_obs.all() <= 0:
        return -np.inf
 
    return np.sum( np.log(p_obs) )
    
def log_probability(theta, x, threshold, Error = False, Bias = False):
    lp = log_prior(theta, threshold)

    if not np.isfinite(lp):
        return -1 * np.inf
    
    Ndet = np.size(x, axis= 0)
    if Bias:
        ls = - Ndet  * log_SelectionBias(theta, threshold)
        
        if not np.isfinite(ls):
            return -1 * np.inf
    else:
        ls = 0

    if Error:
        llh = log_likelihoodWithError(theta, x)
    else :
        llh = log_likelihood(theta, x)

    if not np.isfinite(llh):
        return -np.inf
    
    return lp + ls + llh 

def BiasGenData(x, threshold):
    
    return x[x > threshold ]
## extend data x to include measurement error samples 
def xErrorSampling(x, error_sd, Ns):

    N = np.size(x) 
    x_extended = np.zeros((N, Ns))
    for i in range(N):
        x_extended[i,:] = np.random.normal(x[i],error_sd, Ns)


    return x_extended

def init_position(x,nwalkers,ndim):
    N = np.size( x )
    est_mu = np.sum(x) / N
    est_sd = np.sqrt( np.sum( (x - est_mu)**2  ) / N )
    pos =  np.zeros((nwalkers, ndim))   
    pos[:,0] = np.random.rand(nwalkers) + est_mu
    pos[:,1] = np.random.rand(nwalkers)  + est_sd

    return pos 


def mcmc(x, nwalkers, ndim, threshold, steps, burnt_step, tau_trial, trial_index, Error = False, Bias  = False):
    ## generate initial position on parameter space
    pos = init_position(x,nwalkers,ndim)
    #mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = (x, threshold), kwargs = {'Error' : Error, 'Bias' : Bias})
    # progress = True , just show the progress bar 
    sampler.run_mcmc(pos, steps, progress=True)
    #  pos = (nwalkers,ndim); (:,0) = mu, (:,1) = sd
    samples = sampler.get_chain()

    ## flatten and burn in first x steps 
    ## thin = 15 means only take 1 data every 15 steps
    flat_samples = sampler.get_chain(discard = burnt_step, thin=15, flat=True)
    i = trial_index - 1 
    tau = sampler.get_autocorr_time()
    print('autocorr_time for each parameters=',tau)
    tau_trial[i] = tau.round(3)
    
    return samples, flat_samples
    

def ShowSteps(ndim, trial_index, samples, path): 
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

    figpath = path + '/' +'trial' + str(trial_index) + '_step.png'
    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()
    
def ContourPlot(flat_samples, mu_true, sd_true, trial_index, path):
    
    ### Contour plot
    labels = ["mean", "sd"]
    fig = corner.corner(
        flat_samples, labels=labels, truths=[mu_true, sd_true]
    )

    figpath = path + '/' + 'trial'+str(trial_index) + '_corr.png'
    plt.tight_layout()
    plt.savefig(figpath)
    plt.show()

def SavePercentile(ndim, flat_samples, para_trial, trial_index):
    i = trial_index - 1
    labels = ["mean", "sd"]
    for j in range(ndim):
        mcmc = np.percentile(flat_samples[:, j], [10, 50, 90])
        q = np.diff(mcmc)
        para_trial[j,i,0] = mcmc[1]
        para_trial[j,i,1] = q[0]
        para_trial[j,i,2] =  q[1]
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[j])
        display(Math(txt))



def SaveResult(ndim, tau_trial, para_trial, home):
    
    para_name = ['mean', 'sd']
    for i in range(ndim):   
        f = open(home + '/'+ para_name[i] + '.txt','w')
        np.savetxt(f, para_trial[i])
        f.close()
        #print('output of '+ para_name[i]+ ' is done')

    f = open(home + '/tau.txt','w')
    np.savetxt(f, tau_trial)
    f.close()


def PlotHist(xorigin, x, path, trial_index, Error = False, Bias = False):
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(xorigin, bins = 50)
    axs[0].set_title('x original histogram')
    axs[1].hist(x, bins = 50)
    name = 'x histogram'
    if Error: 
        name += ' Error'
    if Bias:
        name += ' Bias'
    axs[1].set_title(name)

    for ax in axs.flat:
        ax.set(xlabel='x', ylabel='counts')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


    plt.tight_layout()
    figpath = path + '/' + 'trial'+str(trial_index)+ '_corr.png'
    plt.savefig(figpath)
    plt.show()


def createPath(N, mu_true, sd_true, error_sd, Ns, threshold, Error = False, Bias= False):
    ########### Creating path and folders
    home = './mu_sd=' +str(mu_true) + '_' + str(sd_true) + '/' + str(N) + '/'
    
    err_path = 'errSD_Nsample=' + str(error_sd) + '_' + str(Ns) + '/'
    bias_path = 'th=' + str(threshold) + '/'
    if Error and Bias:
        home = home + 'hybird/' + err_path + bias_path

    elif Error:
        home = home + 'err/' + err_path 

    elif Bias:
        home = home + 'bias/' +  bias_path 
    else:
        home = home + 'none/' 
    path_folder = ['/hist', '/steps', '/corr']
    path = [home + string for string in path_folder] 

    for path_x in path:
        if not os.path.exists(path_x):
            os.makedirs(path_x)
    return path,home

def outputPara(home, N, mu_true,sd_true, error_sd, Ns, threshold, nwalkers, burnt_step, trial, Error = False, Bias= False, Seed = False):
    ###Output the parameters in the program 
    
    f = open(home + '/para.txt', 'w')
    f.write('Population =%d' % (N))
    f.write('\nMean=%.1f, SD = %.1f,' % (mu_true, sd_true) )
    if Error:
        f.write('\nError on each data: Yes ')
        f.write('\nError SD for each data = %.1f' % error_sd)
        f.write('\nNumber of samples for each data point =%d' % Ns)
    else:
        f.write('\nError on each data: None ')
    if Bias:

        f.write('\nBias on obs data: Yes ')
        f.write('\nThreshold for the obs data =%.1f' % threshold)

    else:
        f.write('\nBias on obs data: None ')
    
    f.write('\nNumber of walkers in para space =%d' %nwalkers)
    f.write('\nNumber of burnt steps =%d' % burnt_step)
    
    f.write('\nNumber of trial for the run =%d' % trial)
    
    if Seed:
        f.write('\nFixed seed: Yes ')
        f.write('\nSeed of Rand Num =%d' %7)
    else:
        f.write('\nFixed seed =None')

    f.close()

def pclor(mu_true, sd_true, error_sd, Ns, th, mu_low, mu_up, Error = False, Bias = False):
    np.random.Seed(7)
    threshold = mu_true + th * sd_true
    x = np.random.normal(mu_true,sd_true,500)
    #threshold = 0.3
    if Bias:
        x = x[x > threshold]
    N = np.size(x)
    path = './a.png'
    if Error:
        trial_index = 1000
        x = x_samples_assemble(x, error_sd, Ns,trial_index, path)

    mu = np.linspace(mu_low, mu_up,300)
    sd = np.linspace(0.1, 10,300)
    xx,yy = grid = np.meshgrid(mu,sd)

    a = np.zeros((300,300))

    for i in range(300):
        for j in range(300):

            a[i,j] = log_probability(np.array([xx[i,j],yy[i,j]]),N, x, threshold, Error = Error, Bias = Bias)


    index = np. unravel_index(a.argmax(), a.shape)
    #print('hihihihh',index)
    print('max log prob = ',log_probability(np.array([xx[index],yy[index]]),N, x, threshold, Error = Error, Bias = Bias))
    print('mu=',xx[index])
    print('sd=',yy[index])
    print('--------------------------------')
    theta = np.array([mu_true,sd_true])
    print('true para log prob = ',log_probability( theta, N, x, threshold, Error = perr, Bias = psel))
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
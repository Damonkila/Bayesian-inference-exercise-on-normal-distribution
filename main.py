import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import scipy.special as ss
import sys
from IPython.display import display, Math
import function 

def main_p(N, mu_true, sd_true, error_sd, Nsample_for_each_obs, trial, th, err_selec = False, bias_selec = False, seed = False):
    

    #################################################################    Parameters of Population       #####################################################

    ## ndim = 2 for mu and sd 
    ndim = 2 

    #############################################################       Parameters of the model            #########################################################3

    ## number of walkers in the parameter space 
    nwalkers = 30
    ## number of steps burnt out 
    burnt_step = 4000
    steps = 80 * N
    
    threshold = mu_true + th * sd_true
    ##################################################################     MAIN     ####################################################################

    tau_trial = np.zeros((trial, ndim))
    para_trial = np.zeros((2,trial,3))

    ##  Fix seed for random number ON/OFF 
    if seed:
        
        np.random.seed(7)

    path, home = function.createPath(N, mu_true, sd_true, error_sd, Nsample_for_each_obs, threshold, Error = err_selec, Selec_Bias= bias_selec)

    function.outputPara(home, N, mu_true,sd_true, error_sd, Nsample_for_each_obs, threshold, nwalkers, burnt_step, trial, Error = err_selec, Selec_Bias= bias_selec, wseed = seed)

    reader = open(home + '/para.txt', 'r').read()
    print(reader)

    ################## For loop for each trial ######################################

    for i in range(trial):

        trial_index = i + 1
        print('trial=',trial_index)

        ## Generate random data from the population, and histogram

        x = np.random.normal(mu_true, sd_true, N)
    
        figpath = path[0] +'/' +'trial'+str(trial_index)+ '_'+str(N)  + 'hist_normal.png'
        function.plot_hist(N, x, figpath)
        
        ############# Selection Bias on/off by bias_selec 
        if bias_selec:
            x = x[x > threshold]
            figpath = path[0] +'/' +'trial'+str(trial_index)+ '_'+str(N)  + 'hist_bias.png'
            function.plot_bias(N, x,figpath)
            N = np.size(x)

        #### Generate samples after plug in obs error, and plot histogram, switch on/ off by err_selec
        if err_selec:
            x = function.x_samples_assemble(x, error_sd, Nsample_for_each_obs,trial_index, path[0])
            
        ## generate initial position on parameter space
        pos = function.init_position(x,nwalkers,ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, function.log_probability, args = (N, x, threshold), kwargs = {'Error' : err_selec, 'Selec_Bias' : bias_selec})

        # progress = True , just show the progress bar 
        sampler.run_mcmc(pos, steps, progress=True)
        #  pos = (nwalkers,ndim); (:,0) = mu, (:,1) = sd
        samples = sampler.get_chain()
        # show steps of walkers with plots
        function.steps(ndim, trial_index, N, steps, samples, path[1])
        

        tau = sampler.get_autocorr_time()
        print('autocorr_time for each parameters=',tau)
        tau_trial[i] = tau.round(3)
        ## reshape and burn in first 100 steps 
        ## thin = 15 means only take 1 data every 15 steps
        
        flat_samples = sampler.get_chain(discard=burnt_step, thin=15, flat=True)
        #samples2 = sampler.get_chain(discard=100, thin=15)
        labels = ["mean", "sd"]
        function.contour_plot(flat_samples,mu_true, sd_true, N, steps, trial_index, tau_trial[i],labels,path[2])
        
        for j in range(ndim):
            mcmc = np.percentile(flat_samples[:, j], [10, 50, 90])
            q = np.diff(mcmc)
            para_trial[j,i,0] = mcmc[1]
            para_trial[j,i,1] = q[0]
            para_trial[j,i,2] =  q[1]
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(mcmc[1], q[0], q[1], labels[j])
            display(Math(txt))


    para_name = ['mean', 'sd']

    for i in range(ndim):   
        f = open(home + '/'+ para_name[i] + '.txt','w')
        np.savetxt(f, para_trial[i])
        f.close()

        #print('output of '+ para_name[i]+ ' is done')

    f = open(home + '/tau.txt','w')

    np.savetxt(f, tau_trial)

    f.close()


        
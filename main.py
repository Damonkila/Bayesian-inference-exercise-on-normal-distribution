import numpy as np
import function 
import argparse

def main(N, mu_true, sd_true, error_sd, Ns, trial, th, Error = False, Bias = False, Seed = False):

    #################################################################    Parameters of Population       #####################################################
    ## ndim = 2 for mu and sd 
    ndim = 2 
    #############################################################       Parameters of the model            #########################################################3

    ## number of walkers in the parameter space 
    nwalkers = 30
    ## number of steps burnt out 
    burnt_step = 3000
    steps = 50 * N
    threshold = mu_true + th * sd_true
    ##################################################################     MAIN     ####################################################################

    tau_trial = np.zeros((trial, ndim))
    para_trial = np.zeros((2,trial,3))

    ##  Fix Seed for random number ON/OFF 
    if Seed:
        np.random.seed(7)

    path, home = function.createPath(N, mu_true, sd_true, error_sd, Ns, threshold, Error = Error, Bias= Bias)

    function.outputPara(home, N, mu_true,sd_true, error_sd, Ns, threshold, nwalkers, burnt_step, trial, Error = Error, Bias= Bias, Seed = Seed)

    reader = open(home + '/para.txt', 'r').read()
    print(reader)

    ################## For loop for each trial ######################################

    for i in range(trial):
        trial_index = i + 1
        print('trial=',trial_index)

        ## Generate random data from the population, and histogram
        x = np.random.normal(mu_true, sd_true, N)
        xOrigin = x.copy()
        ############# Selection Bias on/off by Bias 
        if Bias:
            x = function.BiasGenData(x, threshold)

        #### Generate samples after plug in obs error, and plot histogram, switch on/ off by Error
        if Error:
            x = function.xErrorSampling(x, error_sd, Ns)
        
        function.PlotHist(xOrigin, x, path[1], trial_index, Error = Error, Bias = Bias)
        # Run Monte Carlo Markov Chain
        samples, flat_samples = function.mcmc(x, nwalkers, ndim, threshold, steps, burnt_step, tau_trial, trial_index, Error = Error, Bias = Bias)
        
        # show steps of walkers with plots
        function.ShowSteps(ndim, trial_index, samples, path[1])
        
        ## Plot covariant plot 
        function.ContourPlot(flat_samples, mu_true, sd_true, trial_index, path[2])
    
        # save the data and error bar from its percentile
        function.SavePercentile(ndim, flat_samples, para_trial, trial_index)

    # Output all result data to files 
    function.SaveResult(ndim, tau_trial, para_trial, home)


        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Population Bayesian inference on Normal distribution.')
    parser.add_argument('--N',type=int,help='Number of events',required=True)
    parser.add_argument('--mu_true',type=float,help='True mean of the population',required=True)
    parser.add_argument('--sd_true',type=float,help='True standard deviation of the population',required=True)
    parser.add_argument('--error_sd',type=float,help='Standard deviation for single event sampling',default=0.2)
    parser.add_argument('--Ns',type=int,help='Number of samples',default=10)
    parser.add_argument('--trial',type=int,help='Number of trials for the same set of parameters',default=1)
    parser.add_argument('--th',type=float,help='threshold = mu_ture + th * sd_true for Selection Bias',default='-0.5')
    parser.add_argument('--Error',type=str,help='True for turning on measurement error effect.',default='False')
    parser.add_argument('--Bias',type=str,help='True for turning on Bias effect.',default='False')
    parser.add_argument('--Seed',type=str,help='True for using fixed seed',default='True')

    args = parser.parse_args()
    N = args.N
    mu_true = args.mu_true
    sd_true = args.sd_true
    error_sd = args.error_sd
    Ns = args.Ns
    trial = args.trial
    th = args.th
    Error = eval(args.Error)
    Bias = eval(args.Bias)
    Seed = eval(args.Seed)

    main(N, mu_true, sd_true, error_sd, Ns, trial, th, Error = Error, Bias = Bias, Seed = Seed)
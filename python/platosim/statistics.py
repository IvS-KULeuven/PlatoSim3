#!/usr/bin/env python3

"""
This python module contains all statistical data analysis utilities 
to more easily analysing PlatoSim data products. 
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot


#--------------------------------------------------------------#
#                       OLS/WLS STATISTICS                     #
#--------------------------------------------------------------#


def hist_fap(hist, fap=0.1):

    """Calculate the False Alarm Probability (PAP) of a histogram.

    hist : ndarray
         Array of histogram counts
    fap : float
        False Alarm Probability (FAP) in percent [%]
    """
    
    wanted_percentiles = [100-fap]
    sx = np.array(hist)
    sx.sort()
    cx = sx.cumsum()
    percentile_sx = [sx[int(len(hist) * p / 100.)] for p in wanted_percentiles]
    percentile_cx = [sx[cx.searchsorted(cx[-1] * p / 100)] for p in wanted_percentiles]

    return percentile_cx





def colortheme(theme):

    """Function to call a color theme.
    """
    
    # Select theme and PI color
    if theme == 'r': color = ['tomato', 'red', 'orange']
    if theme == 'g': color = ['limegreen', 'forestgreen', 'yellowgreen']
    if theme == 'b': color = ['royalblue', 'darkcyan', 'dodgerblue']
    if theme == 'm': color = ['deeppink', 'darkviolet', 'm']
    if theme == 'y': color = ['khaki', 'gold', 'orange']

    return color





def plot_modelfit(data, lsFit, model, lsModel='OLS', CI=[0.05], alpha=0.1, theme='b', 
                  x='x', y='y', xlab='x', ylab='y', yerr=False, figsize=(8,5)):

     
    """Plot OLS/WLS model fit.
    """
   
    # Select theme and PI color
    color = colortheme(theme)
        
    # Set parameter strings 
    reg, pre = 'x', 'y'
    
    # Select model string using R terminology (Linear model is default)
    string = (r'Adj. R$^{2} \approx$ ' + str(round(lsFit.rsquared_adj,2)) + '\n' +
              r'AIC     $\approx$ '    + str(round(lsFit.aic,1)) + '\n' +
              r'BIC     $\approx$ '    + str(round(lsFit.bic,1)) + '\n' +
              r'Cond.  $\approx$ '     + str(round(lsFit.condition_number, 1)) + '\n')
    
    # Fetch fit coefficients
    theta_val, theta_err = [], []
    for i in range(len(lsFit.params)):
        theta_val, theta_err = round(lsFit.params[i],2), round(lsFit.bse[i],2)
        string += fr'$\theta_{i} \approx {theta_val} \pm {theta_err}$' + '\n'
    
    # Select title and model
    title  = r'Fit model: $y = \epsilon + \theta_0$'
    if model == 'y ~ x': title += fr' + $\theta_1 {x}$'
    if model == 'y ~ x + I(x**2)': title += fr' + $\theta_1 {x} + \theta_2 {x}^2$'
    if model == 'y ~ x + I(x**2) + I(x**3)': title += fr' + $\theta_1 {x} + \theta_2 {x}^2 + \theta_3 {x}^3$'
    if model == 'y ~ x + I(x**2) + I(x**3) + I(x**4)':title+=fr' + $\theta_1 {x} + \theta_2 {x}^2 + \theta_3 {x}^3 + \theta_4 {x}^4$'
    if model == 'y ~ x + z': title += fr' + $\theta_1$ {x} + \theta_2 z$'
    if model == 'y ~ x + I(np.sin(x))': title += fr' + $\theta_1 {x} + \theta_2 \sin({x})$'            

    # Predict response variable
    n = 100
    xpredict = np.linspace(data[reg].min(), data[reg].max(), n)
    xpredict = np.linspace(data[reg].min(), data[reg].max(), n)
    
    # Select predictions
    ypredict    = lsFit.predict(exog=dict(x=xpredict))
    predictions = lsFit.get_prediction()
    
    # PLOTTING
    fig, ax = plt.subplots(figsize=figsize)
    
    # plot data
    if yerr:
        ax.errorbar(data[reg], data[pre], yerr=data[yerr], marker='.', color='grey', ls='', alpha=0.1, zorder=0)
        ax.plot(data[reg], data[pre], 'ko', ms=2, alpha=alpha, label="Data", zorder=1)
    else:
        ax.plot(data[reg], data[pre], 'ko', ms=2, alpha=alpha, label="Data", zorder=1)
    
    # Allow more CI intervals
    for i in range(len(CI)):
        df_predictions = predictions.summary_frame(alpha=CI[i])
        df_predictions.index = data.x.values
        # Plot CI
        # ax.fill_between(df_predictions.index, 
        #                 df_predictions.mean_ci_lower, 
        #                 df_predictions.mean_ci_upper, 
        #                 alpha=0.2, color=color[i+1], zorder=2)
        # ax.plot(data[reg], df_predictions.mean_ci_lower, '-', c=color[i+1], lw=1, zorder=2, label=str((1-CI[i])*100)+'\% CI')
        # ax.plot(data[reg], df_predictions.mean_ci_upper, '-', c=color[i+1], lw=1, zorder=2)
    
    # Plot PI -> only relevant for OLS
    ax.fill_between(df_predictions.index, 
                    df_predictions.obs_ci_lower, 
                    df_predictions.obs_ci_upper, 
                    alpha=0.2, color=color[-1], zorder=3)
    ax.plot(data[reg], df_predictions.obs_ci_lower, '-', c=color[-1], lw=1, zorder=2, label=str((1-CI[0])*100)+'\% PI')
    ax.plot(data[reg], df_predictions.obs_ci_upper, '-', c=color[-1], lw=1, zorder=2)
    
    # Plot best fit and data
    ax.plot(df_predictions['mean'], color=color[0], label='Fit', zorder=3)
        
    # Settings
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title,fontsize=15)
    ax.legend(fontsize=13, title='',fancybox=True, framealpha=0.8, loc='upper right')
    ax.set_xlim(data[reg].iloc[0], data[reg].iloc[-1])
    
    # Add fit box
    #RMS = round(np.sqrt(np.mean( (data[pre]-df_predictions['mean'])**2 )))
    #string += f'RMS = {RMS} ppm'
    #props = dict(boxstyle='round', facecolor=color[0], alpha=0.3)
    #ax.text(1.02, 0.98, string, transform=ax.transAxes, fontsize=14, 
    #        verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()
    
    
    
    
def plot_residuals(data, lsFit, reg='x', lsModel='OLS', theme='b', alpha=0.1,
                   figsize=(8,4)):
    
    """Plot the OLS/WLS model fit residuals.
    """
    
    # Choose correct residuals
    
    if lsModel == 'OLS':
        resid = lsFit.resid
    else:
        resid = lsFit.resid_pearson

    # Select the color theme
    
    color = colortheme(theme)

    # Start the plot
    
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot residuals squared vs. observations
    
    ax[0].plot(data[reg], resid**2, 'ko', ms=2, alpha=alpha, zorder=1)
    ax[0].grid(True,   color='grey',   ls='-',  lw=0.5, zorder=2)
    ax[0].axhline(y=0, color=color[0], ls='--', lw=2.0, zorder=3)
    ax[0].set_xlabel(reg)
    ax[0].set_ylabel(r"Residuals squared, $\varepsilon^2$")
    ax[0].set_xlim(data[reg].iloc[0], data[reg].iloc[-1])
    
    # Plot residuals vs. plot
    
    ax[1].plot(lsFit.fittedvalues, resid, 'ko', ms=2, alpha=alpha, zorder=1)
    ax[1].grid(True,   color='grey',    ls='-',  lw=0.5, zorder=2)
    ax[1].axhline(y=0, color=color[0],  ls='--', lw=2.0, zorder=3)
    ax[1].set_xlabel("Predicted reponse")
    ax[1].set_ylabel(r"Residuals, $\epsilon$")
    ax[1].set_xlim(np.min(lsFit.fittedvalues), np.max(lsFit.fittedvalues))

    plt.tight_layout()
    plt.show()
    
    
    

    
def plot_standardized_residuals(data, lsFit, K, reg='x', lsModel='OLS', figsize=(8,7)):

    """Function to plot Q-Q plot.
    """
    
    # Choose correct residuals
    
    if lsModel == 'OLS':
        resid = lsFit.resid
    else:
        resid = lsFit.resid_pearson
    
    N = len(data[reg])
    s2 = np.sum(resid**2) / (N-K)
    standardizedResiduals = resid / np.sqrt(s2)
    
    # Plot standardized residuals (QQ-plot)
    
    fig, ax = plt.subplots(figsize=figsize)
    qqplot(standardizedResiduals, line='45', ax=ax)
    plt.show()





def model_selection(AIC_j, BIC_j, method='BIC', show=False):

    """Perform a model comparison from the AIC and BIC parameters.
    
    The AIC and BIC useful to avoid overfitting. The AIC and BIC
    are essential parameters to penalize the increased complexity
    of a model compared the goodness of fit. The model uses the:
    k : Number of unknown parameters
    N : Number of observations

    Notes
    -----
    Typically the BIC is superior to the AIC when N > 50. Hence,
    as this is true for PLATO light curves we use the BIC as the
    default method to select the best model.
    """

    n = len(AIC_j)
            
    # Akaike and Baysian delta score
    dAIC_j = np.array([AIC_j[i] - np.min(AIC_j) for i in range(n)])
    dBIC_j = np.array([BIC_j[i] - np.min(BIC_j) for i in range(n)])
    
    # Akaike weight and Baysian probability
    wAIC_i = []
    pBIC_i = []
    for i in range(n):    
        wAIC_i.append(np.exp(-1/2*(AIC_j[i]-np.min(AIC_j))) / np.sum(np.exp(-1/2*dAIC_j)))
        pBIC_i.append(np.exp(-1/2*(BIC_j[i]-np.min(BIC_j))) / np.sum(np.exp(-1/2*dBIC_j)))

    # Probabilities
    w12 = wAIC_i[0] / (wAIC_i[0] + wAIC_i[1]) * 100
    p12 = pBIC_i[0] / (pBIC_i[0] + pBIC_i[1]) * 100
    if n > 2:
        w13 = wAIC_i[0] / (wAIC_i[0] + wAIC_i[2]) * 100
        p13 = pBIC_i[0] / (pBIC_i[0] + pBIC_i[2]) * 100
        w23 = wAIC_i[1] / (wAIC_i[1] + wAIC_i[2]) * 100
        p23 = pBIC_i[1] / (pBIC_i[1] + pBIC_i[2]) * 100
    if n > 3:
        w14 = wAIC_i[0] / (wAIC_i[0] + wAIC_i[3]) * 100
        p14 = pBIC_i[0] / (pBIC_i[0] + pBIC_i[3]) * 100
        w24 = wAIC_i[1] / (wAIC_i[1] + wAIC_i[3]) * 100
        p24 = pBIC_i[1] / (pBIC_i[1] + pBIC_i[3]) * 100
        w34 = wAIC_i[2] / (wAIC_i[2] + wAIC_i[3]) * 100
        p34 = pBIC_i[2] / (pBIC_i[2] + pBIC_i[3]) * 100
        
    # Find best model (in %)
    if n == 2:
        # AIC probability
        if w12 >= 50:
            best_aic = 1
        else:
            best_aic = 2
        # BIC probability
        if p12 >= 50:
            best_bic = 1
        else:
            best_bic = 2            
    elif n == 3:
        # AIC probability
        if w12 >= 50 and w13 >= 50:
            best_aic = 1
        elif w12 <= 50 and w23 >= 50:
            best_aic = 2
        else:
            best_aic = 3
        # BIC probability
        if p12 >= 50 and p13 >= 50:
            best_bic = 1
        elif p12 <= 50 and p23 >= 50:
            best_bic = 2
        else:
            best_bic = 3
    elif n == 4:
        # AIC probability
        if w12 >= 50 and w13 >= 50 and w14 >= 50:
            best_aic = 1
        elif w12 <= 50 and w23 >= 50 and w24 >= 50:
            best_aic = 2
        elif w13 <= 50 and w23 <= 50 and w34 >= 50:
            best_aic = 3            
        else:
            best_aic = 4
        # BIC probability
        if p12 >= 50 and p13 >= 50 and p14 >= 50:
            best_bic = 1
        elif p12 <= 50 and p23 >= 50 and p24 >= 50:
            best_bic = 2
        elif p13 <= 50 and p23 <= 50 and p34 >= 50:
            best_bic = 3            
        else:
            best_bic = 4
            
    # Show model output
    if show:
        print('\n------------------------------')
        print('Model comparison')
        for i in range(n):
            print(f'AIC weight model {i+1}: {round(wAIC_i[i], 3)}')
            print(f'BIC proba. model {i+1}: {round(pBIC_i[i], 3)}')
            
        print('\nHeuristical likelihood')
        print(f'w1/w2 : {round(wAIC_i[0]/wAIC_i[1], 4)}')
        print(f'p1/p2 : {round(pBIC_i[0]/pBIC_i[1], 4)}')
        if n > 2:
            print(f'w2/w3 : {round(wAIC_i[1]/wAIC_i[2], 4)}')
            print(f'p2/p3 : {round(pBIC_i[1]/pBIC_i[2], 4)}')
            print(f'w3/w1 : {round(wAIC_i[2]/wAIC_i[0], 4)}')
            print(f'p3/p1 : {round(pBIC_i[2]/pBIC_i[0], 4)}')
        if n > 3:
            print(f'w4/w1 : {round(wAIC_i[3]/wAIC_i[0], 4)}')
            print(f'p4/p1 : {round(pBIC_i[3]/pBIC_i[0], 4)}')
            print(f'w4/w2 : {round(wAIC_i[3]/wAIC_i[1], 4)}')
            print(f'p4/p2 : {round(pBIC_i[3]/pBIC_i[1], 4)}')
            print(f'w4/w3 : {round(wAIC_i[3]/wAIC_i[2], 4)}')
            print(f'p4/p3 : {round(pBIC_i[3]/pBIC_i[2], 4)}')
            
        print('\nProbability in favour of model 1 over 2')
        print(f'w1/(w1+w2) : {round(w12, 1)} %')
        print(f'p1/(p1+p2) : {round(p12, 1)} %')
        if n > 2:
            print('\nProbability in favour of model 1 over 3')
            print(f'w1/(w1+w3) : {round(w13, 1)} %')
            print(f'p1/(p1+p3) : {round(p13, 1)} %')
            print('\nProbability in favour of model 2 over 3')
            print(f'w2/(w2+w3) : {round(w23, 1)} %')
            print(f'p2/(p2+p3) : {round(p23, 1)} %')
        if n > 3:
            print('\nProbability in favour of model 1 over 4')
            print(f'w1/(w1+w4) : {round(w14, 1)} %')
            print(f'p1/(p1+p4) : {round(p14, 1)} %')
            print('\nProbability in favour of model 2 over 4')
            print(f'w2/(w2+w3) : {round(w24, 1)} %')
            print(f'p2/(p2+p3) : {round(p24, 1)} %')
            print('\nProbability in favour of model 3 over 4')
            print(f'w3/(w3+w4) : {round(w34, 1)} %')
            print(f'p3/(p3+p4) : {round(p34, 1)} %')

        print('\n------------------------------')
        print(f'The best AIC model is: {best_aic}')
        print(f'The best BIC model is: {best_bic}')
        print('------------------------------\n')

    if method == 'AIC':
        return best_aic
    elif method == 'BIC':
        return best_bic
    

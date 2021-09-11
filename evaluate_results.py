import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

def save_obj(obj, name, directory='./'):
    with open(directory+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name, directory='./'):
    with open(directory + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
all_rates = load_obj('all_rates', directory='./')

def get_s_and_b(core_1, core_2, r_acc, r_radio, r_fn, r_ibdbg, ncore=1, **kw):
    b = r_acc + r_radio + r_fn + r_ibdbg
    s = core_1
    if ncore==2:
        s+= core_2
    elif ncore==1:
        b+= core_2 
    return s, b

def poisson_time(s, b, 
                Z=3, 
              **kw
                ):
    '''
    Calculate Poisson-based time (days) to anomaly from s (signal) and b (background) 
    Z = significance 
    '''
    bottom = 2 * ((s+b) * np.log((s+b)/b) - s )
    t = Z**2 / bottom
    return t

def get_times(s, b, sigma_b=0, #assume 0 uncertainty if low. 
              Z=3, **kw):
    '''
    Calculate Anomaly and Measurement times from s(signal) and b (background)
    sigma_b=0: Assumes uncertainty is low, since times~1/(b + sigma^2)
    Z = significance 
    '''
    sig_to_bg = s / np.sqrt(b + sigma_b**2)
    
    #measurement assumes Gauss statistics since measurement assumes 
    #counts high enough to be out of regime of poisson
    measurement = (b + s) / (s**2/Z**2 - sigma_b**2)
    
    #anomaly is poisson-like: 
    anomaly = poisson_time(s=s, b=b, Z=Z)
    
    return anomaly, measurement

def ml_roc_rates(acceptances, medium, all_rates):
    '''
    instance: ('core1' or 'core2') key for rates dictionary w/ different cores.
    acceptances: dict with fpr, tpr 
    '''    
    tpr = acceptances['tpr']
    fpr = acceptances['fpr']
    #get rid of 0-multiplication mishaps 
    mask = fpr > 0. 
    fpr, tpr = fpr[mask], tpr[mask]
    
    ml_rates = all_rates[medium].copy()
    for ibdsource in ['core_1', 'core_2', 'r_ibdbg']:
        ml_rates[ibdsource] = ml_rates[ibdsource]*tpr
    for bgsource in ['r_fn']:
        ml_rates[bgsource] = ml_rates[bgsource]*fpr
    
    return ml_rates

def plot_times(acceptances, 
               medium, 
               all_rates, 
               ncore=1,
               extra_title=''):
    
    ml_rates = ml_roc_rates(acceptances, medium, all_rates)
    s, b = get_s_and_b(**ml_rates, ncore=ncore)
    t_anom, t_meas = get_times(s, b)
    
    s0, b0 = get_s_and_b(**all_rates[medium], ncore=ncore)
    t_anom0, t_meas0 = get_times(s0, b0)
    
    tpr = acceptances['tpr']
    fpr = acceptances['fpr']
    #get rid of 0-multiplication mishaps 
    mask = fpr > 0. 
    fpr, tpr = fpr[mask], tpr[mask]
    
    fig, ax = plt.subplots(1,1, figsize=(7, 5), 
#                                   gridspec_kw={'height_ratios': [2, 1]}
                                 )
    plt.suptitle("Dwell Times to Anomaly Detection and Measurement \n %s"%(extra_title))
    ax.hlines(t_anom0, 0, 1, 
           color='royalblue', linestyle='--', lw=2, 
           label='Anomaly (%.1f d)'%(t_anom0)
          )
    ax.hlines(t_meas0, 0, 1, 
           color='darkorange', linestyle='--', lw=2, 
           label='Measurement (%.1f d)'%(t_meas0)
          )
    ax.plot(fpr, t_anom, 
         color='royalblue', lw=2, 
         label='Anomaly w/ ML (%.1f d)'%(np.min(t_anom))
        )

    ax.plot(fpr, t_meas, 
         color='darkorange', lw=2, 
         label='Measurement w/ ML (%.1f d)'%(np.min(t_meas))
        )

    
    ax.legend(ncol=2, 
#               title=r'Dwell Time Calculations', 
              title_fontsize=11)
    ax.set_xlabel('Fast-neutron Acceptance (Rel. to Cuts)', fontsize=12)
    ax.set_ylabel('Dwell time (days)', fontsize=12)    
    ax.grid()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(min([t_anom0, min(t_anom), t_meas0, min(t_meas)])*.8,
             max([t_anom0, t_meas0])*1.2)
    plt.show()
    
def quick_analysis(acceptances, medium, all_rates=all_rates):
    '''
    Quickly returns some plots for the H20 data given a certain roc curve.
    
    acceptances: (type: dict of fpr, tpr)
    '''
    if medium=='h2o': title_extra = r'Gd-H$_2$O'
    if medium=='wbls': title_extra = r'Gd-WbLS'
    
    for ncore in [1,2]:
        extra_title=f'16m {title_extra}, {ncore}-core signal'
        plot_times(acceptances, medium, 
                   all_rates, ncore=ncore, 
                   extra_title=extra_title)
    
# MACHINE LEARNING EVALS
# class MLevals:
#     def __init__(self):
#         return 
def format_outputs(scores, target):
    if scores.shape[1]>1:
        y_pred_2 = np.argmax(scores, axis=1)
        try: target_2 = np.argmax(target, axis=1)
        except: target_2=target
    else:
        y_pred_2, target_2 = np.round(scores), target
    return target_2, y_pred_2

class ML_Evaluation:
    '''
    Feed the true label and probabilistic guesses, 
    to retrieve plots of interest.
    draw_confmat, draw_roc, ...
    '''    
    def __init__(self, y, probs, classes=2):
        self.y = y
        self.probs = probs
        self.classes = classes
        self.preds = format_outputs(probs, y)[1]
    
    def draw_confmat(self, 
                 extra_title='',
                **kwargs):
        confmat = confusion_matrix(self.y, self.preds)
        plt.figure(figsize=(4,4))
        ax = sns.heatmap(confmat, annot=True, fmt='g', cbar=False,
                        )
        ax.set_title('Confusion Matrix fast-n vs. ibd %s'%(extra_title))
        ax.set_xlabel('Predicted PID')
        ax.set_ylabel('Actual PID')

        if self.classes==2: labels = ['fastn', 'ibd']
        elif self.classes==3: labels=['fastn', 'ibd0', 'ibd1']
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.show()

    def draw_roc(self, extra_title='',):
        if self.classes==3: 
            cat1 = [0, 0, 1]
            cat2 = [1, 2, 2]
            labels = ['fastn/ibd0', 'fastn/ibd1', 'ibd0/ibd1',]
        elif self.classes==2:
            cat1, cat2, labels = [0], [1], ['fast-n/ibd']

        plt.figure(figsize=(6,4))
        for i, j, label in zip(cat1, cat2, 
                                labels,
                                 ):
            mask = [any(t) for t in zip(self.y==i, self.y==j)]

            if self.probs.shape[1]>1: prob = self.probs[:, j][mask]
            else: prob = self.probs[mask]
            true = self.y[mask]
            true = true - np.min(true)
            true = true/np.max(true)

            fpr, tpr, thresholds = roc_curve(true, prob)

            auc = roc_auc_score(self.y[mask], prob)
            plt.plot(fpr, tpr, 
                 label='%s (AUC = %0.3f)'%(label, auc), 
                 lw=2, 
                )
        plt.xlim([-0.01, 1.0])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC: fast-n vs. ibd %s'%(extra_title))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.legend()
        plt.grid()
        plt.show()    

        return tpr, fpr, thresholds
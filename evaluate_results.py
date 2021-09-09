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
    
core_rates = load_obj('all_core_rates', directory='./')

def get_times(r_signal, s_signal, 
              r_acc, r_radio, r_fn, r_ibdbg, 
              s_bg, 
              Z=3):
    r_bg = r_acc + r_radio + r_fn + r_ibdbg
    sig_to_bg = r_signal / np.sqrt(r_bg + s_bg**2)
    
    anomaly = r_bg / (r_signal**2/Z**2 - s_bg**2)
    measurement = (r_bg + r_signal) / (r_signal**2/Z**2 - s_bg**2)
    return anomaly, measurement

def calc_new_rates(instance, acceptances, 
                   plot=True, **kwargs
                  ):
    '''
    instance: ('core1' or 'core2') key for rates dictionary w/ different cores.
    acceptances: dict with fpr, tpr 
    '''
    tpr = acceptances['tpr'] 
    fpr = acceptances['fpr']
    mask = fpr > 0  # to get rid of zero-edges
    fpr, tpr = fpr[mask], tpr[mask]

    ml_rates = core_rates[instance].copy()    
    ml_rates['r_signal'] = ml_rates['r_signal']*tpr
    ml_rates['r_fn'] = ml_rates['r_fn']*fpr
    ml_rates['r_ibdbg'] = ml_rates['r_ibdbg']*tpr
    
    anom0, meas0 = get_times(**core_rates[instance])
    anom, meas = get_times(**ml_rates)

    if plot==True:
        plot_times(fpr, anom, meas, meas0, anom0, **kwargs)
        
    return

def plot_times(fpr, anom, meas, meas0, anom0, extra_title=''):
    plt.plot(fpr, anom, 
         color='royalblue', lw=2, 
         label='anomaly w/ ML (%.1f d)'%(np.min(anom))
        )
    plt.hlines(anom0, 0, 1, 
           color='blue', linestyle='--', lw=2, 
           label='anomaly (%.1f d)'%(anom0)
          )
    plt.plot(fpr, meas, 
         color='orange', lw=2, 
         label='measurement w/ ML (%.1f d)'%(np.min(meas))
        )
    plt.hlines(meas0, 0, 1, 
           color='darkorange', linestyle='--', lw=2, 
           label='measurement (%.1f d)'%(meas0)
          )
    
    plt.legend(ncol=2)
    plt.xlabel('Fast-neutron background acceptance')
    plt.ylabel('Dwell time (days)')
    
    plt.grid()
    plt.title("Anomaly Detection and Measurement Dwell Times \n %s"%(extra_title))
    plt.xlim(0.0, 1)
    plt.ylim(min([anom0, min(anom), meas0, min(meas)])*.8,
             max([anom0, meas0])*1.2)

    plt.show()
    
def quick_analysis(acceptances, medium='h20'):
    '''
    Quickly returns some plots for the H20 data given a certain roc curve.
    
    acceptances: (type: dict of fpr, tpr)
    '''
    if medium=='h20': title_extra = 'Gd-H20'
    if medium=='wbls': title_extra = 'Gd-WbLS'
    kw = dict(plot=True, extra_title=f'16m {title_extra}, 1-core signal')
    calc_new_rates(f'core1_{medium}', acceptances, **kw)

    kw = dict(plot=True, extra_title=f'16m {title_extra}, 2-core signal')
    calc_new_rates(f'core2_{medium}', acceptances, **kw)
    
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
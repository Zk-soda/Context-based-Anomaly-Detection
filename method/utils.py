
import torch
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def roc_auc(rec_error, gt_labels, save_roc = False, roc_savepath = None):
    
    ##avoid dev 0
    if torch.min(rec_error) != torch.max(rec_error):
        anomaly_scores = (rec_error - torch.min(rec_error)) / (torch.max(rec_error) - torch.min(rec_error))
    else:
        anomaly_scores = (rec_error - torch.min(rec_error)+1e-7) / (torch.max(rec_error) - torch.min(rec_error)+1e-7)
    ##torch tenor
    anomaly_scores = anomaly_scores.cpu()
    rec_error = rec_error.cpu()
    
    def roc(an_scores, labels, roc_savepath):
        
        # True/False Positive Rates.
        fpr, tpr, _ = roc_curve(labels, an_scores)
        AUC = auc(fpr, tpr)
        
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % (AUC))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        if save_roc == True:
            plt.savefig(roc_savepath)
        plt.show()

        return AUC
    
    AUC = roc(anomaly_scores, gt_labels, roc_savepath)
    
    return AUC

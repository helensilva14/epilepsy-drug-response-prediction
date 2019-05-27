import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, model_name, file_name):
    """Function to plot a confusion matrix without normalization"""
    
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    plt.title('Drug Response Prediction Confusion Matrix without Normalization')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    classNames = ['Refractory','Responsive']
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    
    # set the layout approach without normalization
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j])+" = "+str(cm[i][j]))
            
    # save plot as image 
    plt.savefig('../figures/default-confusion-matrixes/%s-%s-default-cm' % (model_name.lower(), file_name))
    plt.show()    

def plot_normalized_confusion_matrix(cm, model_name, file_name, use_portuguese=False):
    """Function to plot a normalized confusion matrix"""
    
    # apply normalization
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # plot matrix
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # check language to be used
    if(use_portuguese):
        plt.title('Dados genéticos - Matriz de Confusão Normalizada')
        plt.colorbar()
        plt.ylabel('Rótulo verdadeiro') 
        plt.xlabel('Rótulo previsto') 
        classNames = ['Refratário','Responsivo'] 
    else:
        plt.title('Normalized Drug Response Prediction Confusion Matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        classNames = ['Refractory','Responsive']
    
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    
    # set the normalized layout approach
    fmt = '.2f' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), 
                 horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    
    # save plot as image 
    plt.savefig('../figures/normalized-confusion-matrixes/%s-%s-normalized-cm.pdf' % (model_name.lower(), file_name), dpi=300, 
                pad_inches=0, bbox_inches='tight')
    plt.show()

def plot_roc_curve(fpr, tpr, auc_score, model_name, file_name, use_portuguese=False):
    """Generic function to plot a ROC curve"""
    
    plt.figure(1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--') 
    plt.plot(fpr, tpr, color='darkorange', label='AUC = %f)' % auc_score)
    
    # check language to be used
    if(use_portuguese):    
        plt.xlabel('Taxa de falsos positivos')
        plt.ylabel('Taxa de verdadeiros positivos') 
        plt.title('Dados genéticos - Curva ROC') 
    else:
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Drug Response Prediction - ROC Curve')
    
    plt.legend(loc='best')
    # save plot as image 
    plt.savefig('../figures/roc-curves/%s-%s-roc-curve.pdf' % (model_name.lower(), file_name), dpi=300)
    plt.show()
    
def calculate_ecdf(predprobs):
    """Function to calculate the ECDF (empirical cumulative distribution function) for prediction probabilities"""
    
    predprobs = np.sort(predprobs)
    
    percentiles = list()  
    for i in np.arange(1, len(predprobs)+1):
        percentiles.append(i/len(predprobs))
        
    return percentiles

def plot_ecdf(predprobs_responsive, predprobs_refractory):
    """Function to plot the ECDF values vs. Probability Predictions"""
    
    plt.figure(figsize=(10,8))
    plt.plot(np.sort(predprobs_responsive), calculate_ecdf(predprobs_responsive), color='blue', label='Responsive')
    plt.plot(np.sort(predprobs_refractory), calculate_ecdf(predprobs_refractory), color='red', label='Refractory')
    plt.grid(True)
    plt.xlabel('Prediction probabilities')
    plt.ylabel('ECDF')
    plt.title('ECDF for Drug Response Probability Predictions') 
    plt.legend(loc='best')
    plt.savefig('../figures/ecdf/loocv-best-model-genetic-data-ecdf.pdf', dpi=300)
    plt.show()
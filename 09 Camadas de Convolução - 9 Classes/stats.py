import matplotlib.pyplot as plt
import numpy as np

def plotmodelhistory(history):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['train', 'validate'], loc = 'upper left')
    
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['train', 'validate'], loc = 'upper left')
    plt.show()
    fig.savefig('statistics/history.png', dpi=72)

def heatmap(data, row_labels, col_labels, ax = None, cbar_kw = {}, cbarlabel = '', **kwargs):
    
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    
    cbar = ax.figure.colorbar(im, ax= ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = 'bottom')
    
    ax.tick_params(top = True, bottom = False, labeltop = True, labelbottom = False)
    

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    return im, cbar


def annotate_heatmap(im, data = None, fmt = 'd', threshold = None):
    
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment = 'center',
                               color = 'white' if data[i, j] > threshold else 'black')
            
            texts.append(text)
            
    return texts
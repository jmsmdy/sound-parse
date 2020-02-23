import matplotlib.pyplot as plt
import numpy as np

def error(notes_predicted, notes_actual):

    notes_predicted_numpy = 128 * notes_predicted[-1].numpy()
    notes_actual_numpy = 128 * notes_actual[-1].numpy()
        
    fig, axes = plt.subplots(1,3,figsize=(30,10))
        
    #error
    axes[0].imshow(np.log(1+np.abs(notes_actual_numpy - notes_predicted_numpy).T), aspect=(1), cmap='inferno', vmin=np.log(1), vmax=np.log(128))
    axes[0].set_title('Error')
        

    # predicted
    axes[1].imshow(np.log(1+notes_predicted_numpy.T), aspect=(1), cmap='inferno', vmin=np.log(1), vmax=np.log(128))
    axes[1].set_title('Predicted')
        
    #actual
    axes[2].imshow(np.log(1+notes_actual_numpy.T), aspect=(1), cmap='inferno', vmin=np.log(1), vmax=np.log(128))
    axes[2].set_title('Actual')

    return fig
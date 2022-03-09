import matplotlib.pyplot as plt
import os

#TODO also show scaled latent loss after weighting?
def plot_losses(latent_loss, supervision_loss, title, latent_weight = None):
    plt.clf()
    plt.title(title)
    xint = range(0, len(latent_loss))
    plt.xticks(xint)
    plt.yscale('log')
    plt.plot(latent_loss, label='Latent Loss')
    if latent_weight is not None:
        plt.plot([loss * latent_weight for loss in latent_loss], label='Scaled Latent Loss')
    plt.plot(supervision_loss, label='Supervision Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    graphs_folder = 'training_loss_graphs'
    if not os.path.exists(graphs_folder):
        os.makedirs(graphs_folder)



    plt.savefig(graphs_folder + '/Losses_plot_' + title +'.png')

# testing this file
latent_loss = [4.1, 0.4, 0.01, 0.001]
supervision_loss = [0.2, 0.1, 0.12, 0.09]
plot_losses(latent_loss, supervision_loss, "test plot", 1e-1)
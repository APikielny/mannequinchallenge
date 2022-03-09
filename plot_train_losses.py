import matplotlib.pyplot as plt

def plot_losses(latent_loss, supervision_loss, title):
    plt.clf()
    plt.title(title)
    plt.set_yscale('log')
    plt.plot(latent_loss)
    plt.plot(supervision_loss)
    plt.xlabel(Epoch)
    plt.ylabel(Loss)
    plt.savefig('test_plot.png')


latent_loss = [4.1, 0.4, 0.01, 0.001]
supervision_loss = [0.2, 0.1, 0.12, 0.09]
plot_losses(latent_loss, supervision_loss, "test plot")
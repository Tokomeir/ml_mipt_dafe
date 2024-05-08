import matplotlib.pyplot as plt

def plot_losses(train_losses, valid_losess, loss_name, metric_name):
    
    plt.figure(figsize=(6.4 * 2, 4.8 * 1))

    plt.subplot(121)
    plt.plot(train_losses[0, :], color='black', label='train')
    plt.plot(valid_losess[0, :], label='valid')
    plt.xlabel('epoch')
    plt.ylabel(loss_name)
    plt.legend()

    plt.subplot(122)
    plt.plot(train_losses[1, :], color='black', label='train')
    plt.plot(valid_losess[1, :], label='valid')
    plt.xlabel('epoch')
    plt.ylabel(metric_name)
    plt.legend()

    plt.tight_layout()
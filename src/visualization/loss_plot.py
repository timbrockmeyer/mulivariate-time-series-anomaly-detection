import matplotlib.pyplot as plt

def get_loss_plot(train_loss_history, val_loss_history):
    '''
    Returns a pyplot figure object with the plot of the
    train and validation loss lists obtained in training.
    '''
    
    colors = ['#2300a8', '#8400a8'] # '#8400a8', '#00A658'
    plot_dict = {'Training': (train_loss_history, colors[0]), 'Validation': (val_loss_history, colors[1])}
 
    n = len(train_loss_history)
    
    # plot train and val losses and fill area under the curve
    fig, ax = plt.subplots()
    x_axis = list(range(1, n+1))
    for key, (data, color) in plot_dict.items():
        if data is not None:
            ax.plot(x_axis, data, 
                        label=key, 
                        linewidth=2, 
                        linestyle='-', 
                        marker='o', 
                        alpha=1, 
                        color=color)
            ax.fill_between(x_axis, data, 
                        alpha=0.3, 
                        color=color)

    # x axis ticks
    n_x_ticks = 10
    k = max(1, n // n_x_ticks)
    x_ticks = list(range(1, n+1, k))
    ax.set_xticks(x_ticks)

    # figure labels
    ax.set_title('Loss over time', fontweight='bold')
    ax.set_xlabel('Epochs', fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontweight='bold')
    ax.legend(loc='upper right')

    # remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.35, alpha=0.8)

    # log scale of y-axis
    ax.set_yscale('log')

    return fig
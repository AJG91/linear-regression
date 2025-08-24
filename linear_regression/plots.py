
import matplotlib.pyplot as plt

def plot_projectile_motion(t, x, y, path, dpi):
    
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(8, 10))

    ax1.scatter(t, x, label='x')
    ax1.scatter(t, y, label='y')
    ax1.set_xlabel('$t$ [s]', fontsize=14)
    ax1.set_ylabel('position [m]', fontsize=14)
    ax1.legend()
    
    ax2.scatter(x, y)
    ax2.set_xlabel('$x$ [m]', fontsize=14)
    ax2.set_ylabel('$y$ [m]', fontsize=14)
    
    plt.show();
    
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01)
    fig.savefig(path + 'proj_motion_plot.png', bbox_inches='tight', dpi=dpi)

    return None

def plot_true_vs_pred(x, y, y_pred):
    
    plt.scatter(x, y, label='true')
    plt.scatter(x, y_pred, label='pred')
    
    plt.xlabel('Indep. variable', fontsize=14)
    plt.ylabel('Dep. variable', fontsize=14)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

    return None
    
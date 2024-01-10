import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from tqdm.auto import tqdm

def plot_l96_traj(
    x,
    model,
    linewidth,
):
    fig = plt.figure(figsize=(linewidth, linewidth/3))
    plt.grid(False)
    im = plt.imshow(
        x.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, (model.dt/model.lyap_time)*x.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    plt.colorbar(im)
    plt.xlabel('Time (Lyapunov time)')
    plt.ylabel('Lorenz 96 variables')
    plt.tick_params(direction='out', left=True, bottom=True)
    plt.show()

def plot_l96_compare_traj(
    x_ref,
    x_pred,
    model,
    linewidth,
):
    error = x_pred - x_ref
    fig = plt.figure(figsize=(linewidth, linewidth))
    ax = plt.subplot(311)
    ax.grid(False)
    im = plt.imshow(
        x_ref.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, (model.dt/model.lyap_time)*x_pred.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    ax.set_title('true model integration')
    plt.colorbar(im)
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    ax.set_xticklabels([])
    ax = plt.subplot(312)
    ax.grid(False)
    im = plt.imshow(
        x_pred.T,
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 60, as_cmap=True),
        extent = [0, (model.dt/model.lyap_time)*x_pred.shape[0], 0, model.Nx],
        vmin = -10,
        vmax = 15,
    )
    ax.set_title('surrogate model integration')
    plt.colorbar(im)
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    ax.set_xticklabels([])
    ax = plt.subplot(313)
    ax.grid(False)
    im = ax.imshow(
        error.T, 
        aspect = 'auto',
        origin = 'lower',
        interpolation = 'spline36',
        cmap = sns.diverging_palette(240, 10, as_cmap=True),
        extent = [0, (model.dt/model.lyap_time)*error.shape[0], 0, model.Nx],
        vmin = -15,
        vmax = 15,
    )
    ax.set_title('signed error')
    plt.colorbar(im)
    ax.set_xlabel('Time (Lyapunov time)')
    ax.set_ylabel('Lorenz 96 variables')
    ax.tick_params(direction='out', left=True, bottom=True)
    plt.show()

def get_plotly_color_palette(alpha=None):
    if alpha is None:
        return [
            'rgb(99, 110, 250)',
            'rgb(239, 85, 59)',
            'rgb(0, 204, 150)',
            'rgb(171, 99, 250)',
            'rgb(255, 161, 90)',
            'rgb(25, 211, 243)',
            'rgb(255, 102, 146)',
            'rgb(182, 232, 128)',
            'rgb(255, 151, 255)',
            'rgb(254, 203, 82)'
        ]
    else:
        return [
            f'rgba(99, 110, 250, {alpha})',
            f'rgba(239, 85, 59, {alpha})',
            f'rgba(0, 204, 150, {alpha})',
            f'rgba(171, 99, 250, {alpha})',
            f'rgba(255, 161, 90, {alpha})',
            f'rgba(25, 211, 243, {alpha})',
            f'rgba(255, 102, 146, {alpha})',
            f'rgba(182, 232, 128, {alpha})',
            f'rgba(255, 151, 255, {alpha})',
            f'rgba(254, 203, 82, {alpha})'
        ]

def plot_l96_forecast_skill(
    fss,
    model,
    p1,
    p2,
    xmax,
    linewidth,
):
    fig = go.Figure()
    palette = get_plotly_color_palette()
    spalette = get_plotly_color_palette(alpha=0.2)
    
    for (index, key) in enumerate(fss):
        
        time = (model.dt/model.lyap_time)*np.arange(fss[key].shape[0])
        rmse_m = fss[key].mean(axis=1) / model.model_var
        rmse_p1 = np.percentile(fss[key], p1, axis=1) / model.model_var
        rmse_p2 = np.percentile(fss[key], p2, axis=1) / model.model_var
        
        fig.add_scatter(
            x=time,
            y=rmse_m,
            name=key,
            customdata=np.arange(len(time)),
            hovertemplate='index = %{customdata}, value = %{y:.3f}',
            line_color=palette[index]
        )
        fig.add_scatter(
            x=np.concatenate([time, time[::-1]]),
            y=np.concatenate([rmse_p1, rmse_p2[::-1]]),
            fill='toself',
            name=key+' (CI)',
            hoverinfo='skip',
            fillcolor=spalette[index],
            line_width=0,
            mode='lines'
        )        
        
    fig.update_xaxes(title_text='Time (Lyapunov time)')
    fig.update_yaxes(title_text='normalised RMSE')
    fig.update_layout(
        title='Forecast skill', 
        xaxis_range=[0, xmax], 
        yaxis_range=[0, 2], 
        width=linewidth, 
        height=0.7*linewidth,
        hovermode='x unified',
    )
    fig.add_hline(
        y=np.sqrt(2), 
        line_width=1,
        line_dash='dash',
        line_color='black',
        label_text=r'$\sqrt{2}$',
        label_textposition='start',
    )
    fig.show()
    
def plot_learning_curve(
    loss,
    val_loss,
    title,
    linewidth,
):
    
    fig = go.Figure()
    palette = get_plotly_color_palette()
    
    fig.add_scatter(
        x=np.arange(len(loss)),
        y=loss,
        name='training loss',
        customdata=np.arange(len(loss)),
        hovertemplate='epoch = %{customdata}, value = %{y:.3f}',
        line_color=palette[0]
    )
    
    fig.add_scatter(
        x=np.arange(len(val_loss)),
        y=val_loss,
        name='validation loss',
        customdata=np.arange(len(val_loss)),
        hovertemplate='epoch = %{customdata}, value = %{y:.3f}',
        line_color=palette[1]
    )
    
    fig.update_xaxes(title_text='Number of epochs')
    fig.update_yaxes(title_text='MSE', type='log')
    fig.update_layout(title=title, width=linewidth, height=0.7*linewidth, hovermode='x unified')

    fig.show()

class TQDMCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, desc, loss=None, val_loss=None):
        super().__init__()
        self.desc = desc
        self.metrics = {'loss':loss, 'val_loss':val_loss}
    
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.params['epochs'], desc=self.desc)
    
    def on_train_end(self, logs=None):
        self.epoch_bar.close()
        
    def on_epoch_end(self, epoch, logs=None):
        for name in self.metrics:
            self.metrics[name]  = logs.get(name, self.metrics[name])
        self.epoch_bar.set_postfix(mse=self.metrics['loss'], val_mse=self.metrics['val_loss'], refresh=False)
        self.epoch_bar.update()
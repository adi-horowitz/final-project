
import torch
from plot import plot_fit


def print_fit_res(model_name: str):
    checkpoint_filename = f'{model_name}.pt'
    saved_state = torch.load(checkpoint_filename,
                             'cpu')
    fit_res = saved_state['fit_result']
    plot_fit(fit_res)


if __name__ == '__main__':
    model_name = input('insert model name')
    print_fit_res(model_name)

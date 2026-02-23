import matplotlib.pyplot as plt
import torch
import numpy as np
from torchcfm.utils import sample_8gaussians
from utils.otcfm import (sample_conditional_pt, sample_gaussian)
from utils.mean_cfm import get_full_velocity_field



def plot_cos_sim_time(x0, x1, t_list, n_pts=256, sigmamin=0):
    """
    histogram of cosine similarity between the conditional and optimal velocity fields, for all times in t_list
    """
    u_cond = x1 - x0
    _, axes = plt.subplots(1, 6, figsize=(18, 3))

    for ax, t in zip(axes, t_list):
        with torch.no_grad():
            t_vect = torch.ones(n_pts, 1) * t
            xt = sample_conditional_pt(x0, x1, t_vect, sigma=sigmamin)

            u_star = get_full_velocity_field(t_vect, xt, x1, sigmamin=sigmamin)

            cos = torch.nn.functional.cosine_similarity(u_star, u_cond, dim=-1).numpy()
            cos = np.round(cos, 1)

            unique_cos, counts = np.unique(cos, return_counts=True)
            counts = counts / counts.sum()

            ax.bar(unique_cos, counts, width=0.1, edgecolor='black')
            ax.set_title("$t =$" + str(t))
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([0, 1])

    plt.show()


def plot_cos_sim_dimension(dims, n_pts=256):
    """
    histogram of cosine similarity between random vector pairs from a gaussian, for all dimensions in dims
    """
    _, axes = plt.subplots(1, 6, figsize=(18, 3))

    for ax, dim in zip(axes, dims):
        rand_vect = torch.randn(n_pts, dim)
        ref_vect = rand_vect[0:1].repeat(n_pts-1, 1)
        
        cos = torch.nn.functional.cosine_similarity(rand_vect[1:], ref_vect).numpy()
        cos = np.round(cos, 1)
        
        unique_cos, counts = np.unique(cos, return_counts=True)
        counts = counts / counts.sum()
        
        ax.bar(unique_cos, counts, width=0.1, edgecolor='black')
        ax.set_title("d =" + str(dim))
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([0, 1])

    plt.show()



if __name__ == "__main__":
    n_pts = 256

    sigmamin = 0
    t_list = [0.01, 0.21, 0.40, 0.60, 0.79, 0.99]
    x0 = sample_gaussian(n_pts)
    x1 = sample_8gaussians(n_pts)
    plot_cos_sim_time(x0, x1, t_list, n_pts, sigmamin)


    dims = [2, 3, 20, 300, 2000, 3072]
    plot_cos_sim_dimension(dims, n_pts)

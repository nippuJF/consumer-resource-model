# Step 1: Install & Import
!pip install pandas numpy scipy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import least_squares

# Step 2: Upload CSVs
from google.colab import files
uploaded = files.upload()

# Step 3: Load Data (modify as needed)
G14_microbiome = pd.read_csv("G14_microbiome.csv", index_col=0).clip(lower=1e-6, upper=1e2)
G14_metabolomics = pd.read_csv("G14_metabolomics.csv", index_col=0).clip(lower=1e-6, upper=1e2)

# Step 4: Define CR Model
def cr_model(X, t, c, y, d, s, w):
    N, R = X[:len(c)], X[len(c):]
    dNdt = N * (np.dot(y * c, R) - d)
    dRdt = s - w * R - np.sum((c * N[:, None]) * R[None, :], axis=0)
    return np.concatenate([dNdt, dRdt])

def fit_cr_model(microbiome_df, metabolomics_df):
    species = microbiome_df.columns.tolist()
    resources = metabolomics_df.columns.tolist()
    N0 = microbiome_df.iloc[0].values
    R0 = metabolomics_df.iloc[0].values
    X0 = np.concatenate([N0, R0])
    n_species, n_resources = len(species), len(resources)
    t_points = microbiome_df.index.astype(float).values
    target_all = np.concatenate([microbiome_df.values.flatten(), metabolomics_df.values.flatten()])

    def simulate(params):
        idx = 0
        c = params[idx:idx + n_species * n_resources].reshape(n_species, n_resources); idx += n_species * n_resources
        y = params[idx:idx + n_species * n_resources].reshape(n_species, n_resources); idx += n_species * n_resources
        d = params[idx:idx + n_species]; idx += n_species
        w = params[idx:idx + n_resources]
        s = np.zeros(n_resources)
        try:
            sol = odeint(cr_model, X0, t_points, args=(c, y, d, s, w), hmax=1.0)
            return np.concatenate([sol[:, :n_species].flatten(), sol[:, n_species:].flatten()])
        except:
            return np.full_like(target_all, 1e6)

    np.random.seed(42)
    init_params = np.concatenate([
        np.random.rand(n_species * n_resources) * 1e-5,
        np.random.rand(n_species * n_resources) * 0.1,
        np.random.uniform(0.005, 0.05, n_species),
        np.random.uniform(0.005, 0.02, n_resources)
    ])

    result = least_squares(lambda p: simulate(p) - target_all, init_params, method='trf', max_nfev=5000)
    params = result.x

    c = params[:n_species * n_resources].reshape(n_species, n_resources)
    y = params[n_species * n_resources:2 * n_species * n_resources].reshape(n_species, n_resources)
    d = params[2 * n_species * n_resources:2 * n_species * n_resources + n_species]
    w = params[-n_resources:]
    s = np.zeros(n_resources)
    t_sim = np.linspace(t_points[0], t_points[-1], 300)
    sim_result = odeint(cr_model, X0, t_sim, args=(c, y, d, s, w), hmax=1.0)
    species_df = pd.DataFrame(sim_result[:, :n_species], columns=species, index=t_sim)
    resources_df = pd.DataFrame(sim_result[:, n_species:], columns=resources, index=t_sim)
    return species_df, resources_df, (c, y, d, w)

# Step 5: Fit Model
G14_species_fit, G14_resources_fit, G14_params = fit_cr_model(G14_microbiome, G14_metabolomics)

import os
import numpy as np

def graph_defaults():
    import warnings
    import seaborn as sns
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore")
    sns.set_theme()
    sns.set(rc={'figure.figsize':(16,10)}, style='white')
    plt.figure(dpi=600)

def fit_cv(reg, data, model):
    import warnings
    warnings.filterwarnings("ignore")

    cwd = os.getcwd()
    model_path = f"{cwd}/data/{model}.npy"
    params = None

    if os.path.exists(model_path):
        _params = np.load(model_path, allow_pickle=True)
        params = dict(_params.item())
    else:
        X, y = data
        reg.fit(X, y)
        params = reg.best_params_
        np.save(model_path, params)

    return params

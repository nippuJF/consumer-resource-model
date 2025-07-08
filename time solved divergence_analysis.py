# Step 1: Install dependencies
!pip install tslearn numpy pandas matplotlib scipy

# Step 2: Upload the four ZIP files
from google.colab import files
uploaded = files.upload()

import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.integrate import simpson

# Step 3: Extract ZIP files
def extract_zip(zip_name):
    folder = zip_name.replace(".zip", "")
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(folder)
    return folder

folders = {}
for zip_file in uploaded:
    folders[zip_file.replace(".zip", "")] = extract_zip(zip_file)

# Step 4: Load relevant resource dynamics
def load_resource(folder, fname):
    path = os.path.join(folder, fname)
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(float)
    return df

datasets = {
    'G20_feces_14028': load_resource(folders['G20_feces'], 'resources_G14_feces.csv'),
    'G20_feces_SE': load_resource(folders['G20_feces'], 'resources_GE_feces.csv'),
    'G20_pure_14028': load_resource(folders['G20_pure'], 'resources_G14_pure.csv'),
    'G20_pure_SE': load_resource(folders['G20_pure'], 'resources_GE_pure.csv'),
    'L8_feces_14028': load_resource(folders['L8_feces'], 'resources_fit_L14.csv'),
    'L8_feces_SE': load_resource(folders['L8_feces'], 'resources_fit_LE.csv'),
    'L8_pure_14028': load_resource(folders['L8_pure'], 'resources_fit_pure_L14.csv'),
    'L8_pure_SE': load_resource(folders['L8_pure'], 'resources_fit_pure_LE.csv')
}

# Step 5: Interpolate onto common time grid
common_time = np.linspace(0, 72, 300)
def interpolate_df(df):
    return pd.DataFrame(
        np.array([np.interp(common_time, df.index, df[col]) for col in df.columns]).T,
        columns=df.columns, index=common_time
    )

interp_data = {k: interpolate_df(v) for k, v in datasets.items()}

# Step 6: Compute divergence metrics
def compute_divergence(df1, df2):
    shared = df1.columns.intersection(df2.columns)
    mean_diff = np.abs(df1[shared].mean() - df2[shared].mean())
    auc_diff = np.abs(
        simpson(df1[shared].values, x=common_time, axis=0) -
        simpson(df2[shared].values, x=common_time, axis=0)
    )
    dtw_diff = pd.Series({col: dtw(df1[col], df2[col]) for col in shared})
    return mean_diff, auc_diff, dtw_diff

pairs = {
    'G20_feces': ('G20_feces_14028', 'G20_feces_SE'),
    'G20_pure': ('G20_pure_14028', 'G20_pure_SE'),
    'L8_feces': ('L8_feces_14028', 'L8_feces_SE'),
    'L8_pure': ('L8_pure_14028', 'L8_pure_SE')
}

results = {}
for label, (s1, s2) in pairs.items():
    results[label] = compute_divergence(interp_data[s1], interp_data[s2])

# Step 7: Export results
os.makedirs("outputs", exist_ok=True)
for k, (mean, auc, dtw_) in results.items():
    mean.to_csv(f"outputs/{k}_mean.csv")
    pd.Series(auc, index=mean.index).to_csv(f"outputs/{k}_auc.csv")
    dtw_.to_csv(f"outputs/{k}_dtw.csv")

# Step 8: Plot divergence summary
for method, idx in zip(['Mean', 'AUC', 'DTW'], range(3)):
    plt.figure(figsize=(10,6))
    for label, res in results.items():
        if isinstance(res[idx], np.ndarray):
            to_plot = pd.Series(res[idx], index=res[0].index)  
        else:
            to_plot = res[idx]

        to_plot.sort_values(ascending=False).head(10).plot.bar(label=label)

    plt.title(f"Top Metabolite Divergence - {method}")
    plt.ylabel(f"{method} Divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/top_{method}_divergence.png", dpi=300)
    plt.show()

# Step 9: Bundle ZIP
zipfile_name = "CR_model_divergence_results.zip"
with zipfile.ZipFile(zipfile_name, "w") as zipf:
    for file in os.listdir("outputs"):
        zipf.write(os.path.join("outputs", file), arcname=file)

files.download(zipfile_name)

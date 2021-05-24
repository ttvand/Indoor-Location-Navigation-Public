import random
import os
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, callbacks
from tensorflow.python.keras.utils.vis_utils import plot_model

from utils import get_loss
from utils import TEST_SITES
from utils import get_data_folder

N_SPLITS = 3
SITE_IDX = None
N_TOP_BSSIDS = 20
BATCH_SIZE = 128
OVERWRITE = True

data_folder = get_data_folder()
summary_path = data_folder / "file_summary.csv"
stratified_holdout_path = data_folder / "holdout_ids.csv"

# Using https://www.kaggle.com/hiro5299834/indoor-navigation-and-location-wifi-features
pivot_data_folder = data_folder / "pivot"

holdout_df = pd.read_csv(stratified_holdout_path)
if SITE_IDX is None:
  sites = TEST_SITES
  pivot_paths = [pivot_data_folder / f"{site}_train.csv" for site in sites]
  holdout_df = holdout_df[holdout_df["test_site"]]
else:
  analysis_site = TEST_SITES[SITE_IDX]
  sites = [analysis_site]
  holdout_df = holdout_df[holdout_df["site_id"] == analysis_site]
  pivot_paths = [pivot_data_folder / f"{analysis_site}_train.csv"]

train_path = pivot_data_folder / "train_all.csv"
valid_path = pivot_data_folder / "valid_all.csv"

train_trajectories = holdout_df[~holdout_df["holdout"]]["ext_path"].tolist()
train_trajectories = [Path(path).stem for path in train_trajectories]

valid_trajectories = holdout_df[holdout_df["holdout"]]["ext_path"].tolist()
valid_trajectories = [Path(path).stem for path in valid_trajectories]


def set_seed(seed=42):
  random.seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


def write_input_csv(pivot_paths_, valid_, overwrite_):
  path = valid_path if valid_ else train_path
  if path.exists() and not overwrite_:
    print(path, "already exists.")
    return

  data = {
      **{f"bssid_{i}": [] for i in range(N_TOP_BSSIDS)},
      **{f"rssi_{i}": [] for i in range(N_TOP_BSSIDS)},
      **{
          "site": [],
          "x": [],
          "y": [],
          "path": [],
      },
  }
  for p in pivot_paths_:
    with pd.read_csv(p, index_col=0, chunksize=4096) as reader:
      for df in reader:
        site = p.stem.split("_")[0]
        if valid_:
          df = df[df["path"].isin(valid_trajectories)]
        else:
          df = df[df["path"].isin(train_trajectories)]

        for _, row in df.iterrows():
          features = (
              row[:-4].astype(int).sort_values(
                  ascending=False).head(N_TOP_BSSIDS))

          for i in range(N_TOP_BSSIDS):
            data[f"bssid_{i}"].append(features.index[i])
            data[f"rssi_{i}"].append(features[i])
          data["site"].append(site)
          data["x"].append(row[-4])
          data["y"].append(row[-3])
          data["path"].append(row[-1])

  out_df = pd.DataFrame(data=data)
  out_df.to_csv(path, index=False)


write_input_csv(pivot_paths, valid_=True, overwrite_=OVERWRITE)
write_input_csv(pivot_paths, valid_=False, overwrite_=OVERWRITE)


def create_model(n_unique_bssids, n_unique_sites):
  # shared layers
  bssid_embed = layers.Embedding(n_unique_bssids, 16)
  rssi_dense = layers.Dense(64, activation="relu")

  inputs_bssid = []
  inputs_rssi = []
  xs_bssid = []
  xs_rssi = []
  for i in range(N_TOP_BSSIDS):
    input_bssid_layer = layers.Input(shape=(1,))
    x1 = bssid_embed(input_bssid_layer)
    x1 = layers.Flatten()(x1)
    inputs_bssid.append(input_bssid_layer)
    xs_bssid.append(x1)

    input_rssi_layer = layers.Input(shape=(1,))
    x2 = layers.BatchNormalization()(input_rssi_layer)
    x2 = rssi_dense(x2)
    inputs_rssi.append(input_rssi_layer)
    xs_rssi.append(x2)

  input_site_layer = layers.Input(shape=(1,))
  x3 = layers.Embedding(n_unique_sites, 2)(input_site_layer)
  x3 = layers.Flatten()(x3)

  inputs = []
  inputs.extend(inputs_bssid)
  inputs.extend(inputs_rssi)
  inputs.append(input_site_layer)

  xs = []
  xs.extend(xs_bssid)
  xs.extend(xs_rssi)
  xs.append(x3)

  x = layers.Concatenate(axis=1)(xs)

  x = layers.BatchNormalization()(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(256, activation="relu")(x)

  x = layers.Reshape((1, -1))(x)
  x = layers.BatchNormalization()(x)
  x = layers.Bidirectional(
      layers.LSTM(
          128,
          dropout=0.3,
          recurrent_dropout=0.3,
          return_sequences=True,
          activation="relu",
      ))(
          x)
  x = layers.Bidirectional(
      layers.LSTM(16, dropout=0.1, return_sequences=False, activation="relu"))(
          x)

  output_layer_1 = layers.Dense(2, name="xy")(x)
  # output_layer_2 = layers.Dense(1, activation='softmax', name='floor')(x)

  model_ = tf.keras.Model(inputs, [output_layer_1])  # , output_layer_2])

  model_.compile(
      optimizer=tf.optimizers.Adam(lr=0.001), loss="mse", metrics=["mse"])

  return model_


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(valid_path)

bssids = np.concatenate((
    train_df.iloc[:, :N_TOP_BSSIDS].values.reshape(-1),
    test_df.iloc[:, :N_TOP_BSSIDS].values.reshape(-1),
))
unique_bssids = np.unique(bssids)
le_bssid = LabelEncoder()
le_bssid.fit(unique_bssids)

le_site = LabelEncoder()
le_site.fit(np.array(sites, dtype=str))

rssis = np.concatenate((
    train_df.iloc[:, N_TOP_BSSIDS:2 * N_TOP_BSSIDS].values,
    test_df.iloc[:, N_TOP_BSSIDS:2 * N_TOP_BSSIDS].values,
))
ss = StandardScaler()
ss.fit(rssis)

model = create_model(
    n_unique_bssids=len(unique_bssids),
    n_unique_sites=len(sites),
)
# plot_model(model)


def prepro(df):
  for i in range(N_TOP_BSSIDS):
    df.iloc[:, i] = le_bssid.transform(df.iloc[:, i])
  df.iloc[:, N_TOP_BSSIDS:2 * N_TOP_BSSIDS] = ss.transform(
      df.iloc[:, N_TOP_BSSIDS:2 * N_TOP_BSSIDS].values)
  df.iloc[:, -4] = le_site.transform(df.iloc[:, -4])
  return df

train_df = prepro(train_df)
test_df = prepro(test_df)


def reshape_x(df):
  out = []
  out.extend([df.iloc[:, i] for i in range(N_TOP_BSSIDS)])
  out.extend([df.iloc[:, N_TOP_BSSIDS + i] for i in range(N_TOP_BSSIDS)])
  out.append(df.iloc[:, -3])
  return tuple(out)


def reshape_y(df):
  return np.stack((df.iloc[:, -2], df.iloc[:, -1]), axis=1)


(data_folder / "models").mkdir(exist_ok=True)

test_df = test_df.iloc[:, :-1]

fold = 0
for fold, (train_idx, val_idx) in enumerate(
    StratifiedKFold(n_splits=3, shuffle=True).split(train_df.iloc[:, -1],
                                                    train_df.iloc[:, -1])):
  train_data = train_df.iloc[train_idx, :-1]
  valid_data = train_df.iloc[val_idx, :-1]
  model.fit(
      reshape_x(train_data),
      y=reshape_y(train_data),
      validation_data=(reshape_x(valid_data), reshape_y(valid_data)),
      epochs=1000,
      callbacks=[
          callbacks.ReduceLROnPlateau(
              monitor="val_loss",
              factor=0.1,
              patience=3,
              verbose=1,
              min_delta=1e-4,
              mode="min",
          ),
          callbacks.ModelCheckpoint(
              data_folder / "models" / f"{fold}_rnn.hdf5",
              monitor="val_loss",
              verbose=0,
              save_best_only=True,
              save_weights_only=True,
              mode="min",
          ),
          callbacks.EarlyStopping(
              monitor="val_loss",
              min_delta=1e-4,
              patience=5,
              mode="min",
              baseline=None,
              restore_best_weights=True,
          ),
      ],
  )

  model.load_weights(data_folder / "models" / f"{fold}_rnn.hdf5")
  print(f"{fold}: mpe: {get_loss(model.predict(reshape_x(valid_data)), reshape_y(valid_data))}")

model.load_weights(data_folder / "models" / f"{fold}_rnn.hdf5",)
preds = model.predict(reshape_x(test_df))
print(f"final mpe: {get_loss(preds, reshape_y(test_df))}")

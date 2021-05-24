import datetime
import gc
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
import time

import utils

mode = ['train', 'test'][1]
override_model_ext = [None, 'train - 2021-04-08 16:30:09'][0]
only_process_test_sites = True
store_predictions = True

sensor_group_cols = ['acce', 'gyro', 'ahrs']
models_group_name = 'sensor_distance'
data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)
sensor_folder = data_folder / 'sensor_data'
save_ext = '' if only_process_test_sites else '_all_sites'
model_folder = data_folder.parent / 'Models' / models_group_name

if not 'test' in locals() or not 'loaded_mode' in locals() or (
    mode != loaded_mode) or not 'sensor_cols' in locals():
  test_data_path = data_folder / 'test' / (
    '0a5c442ffa2664e89e18262d_reshaped.pickle')
  with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)
  all_sensor_cols = test_data['shared_time'].columns.tolist()
  sensor_cols = [c for c in all_sensor_cols if any(
    [g in c for g in sensor_group_cols]) and not 'uncali' in c and (
      not c[:2] == 'a_')]
  
  with open(sensor_folder / ('train' + save_ext + '.pickle'), 'rb') as f:
    train = pickle.load(f)
  with open(sensor_folder / ('valid' + save_ext + '.pickle'), 'rb') as f:
    valid = pickle.load(f)
  
  if mode == 'train':
    predict = ('valid', valid)
  else:
    train.update(valid)
    valid = None
    with open(sensor_folder / ('test' + save_ext + '.pickle'), 'rb') as f:
      test = pickle.load(f)
    predict = ('test', test)
    
  predict = predict if store_predictions else None
    
  loaded_mode = mode

class ZDataset:
  def __init__(
      self, sub_trajectory_keys, raw_data, samples_per_epoch, sensor_cols,
      ser_limit, fixed_order):
    self.ser_limit = ser_limit
    self.fixed_order = fixed_order
    self.sample_id = 0
    n = len(sub_trajectory_keys)
    n_inputs = len(sensor_cols)
    
    sensor_data = np.zeros((n, ser_limit, n_inputs), dtype=np.float16)
    target_distances = np.zeros(n, dtype=np.float32)
    num_non_padded = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
      # print(i)
      k, r = sub_trajectory_keys[i]
      sub_traj_data = raw_data[k]['waypoint_segments'][r][sensor_cols].values
      raw_val_count = sub_traj_data.shape[0]
      sensor_data[i, :raw_val_count] = sub_traj_data
      num_non_padded[i] = raw_val_count
      if raw_data[k]['relative_waypoint_distances'] is not None:
        target_distances[i] = raw_data[k]['relative_waypoint_distances'][r]
      else:
        target_distances[i] = -1
      
    self.samples_per_epoch = samples_per_epoch
    self.n = n
    self.sensor_data = torch.from_numpy(sensor_data)
    self.target_distances = torch.from_numpy(target_distances)
    self.num_non_padded = num_non_padded

  def __len__(self):
    return self.samples_per_epoch
  
  def __getitem__(self, idx):
    if self.fixed_order:
      i1 = self.sample_id
      self.sample_id += 1
    else:
      i1 = np.random.randint(self.n)
    recurrent_last_step_mask = torch.zeros(self.ser_limit, dtype=torch.bool)
    recurrent_last_step_mask[self.num_non_padded[i1]-1] = True
    return {
        'sensor_obs': self.sensor_data[i1],
        'recurrent_last_step_mask': recurrent_last_step_mask,
        'y': self.target_distances[i1],
    }

class ZNN(nn.Module):
  def __init__(
      self, ser_limit, n_units, n_inputs, num_recurrent_layers,
      bidirectional_rnn):
    super(ZNN, self).__init__()
    self.ser_limit = ser_limit
    self.n_units = n_units
    self.n_inputs = n_inputs
    self.num_recurrent_layers = num_recurrent_layers
    self.bidirectional_rnn = bidirectional_rnn
    
    n_recurrent_units = n_units//2 if bidirectional_rnn else n_units
    self.rec = nn.GRU(
      input_size=n_inputs, hidden_size=n_recurrent_units,
      num_layers=num_recurrent_layers, dropout=0, batch_first=True,
      bidirectional=bidirectional_rnn)
    
    self.lin = nn.Sequential(
        nn.Linear(n_units, n_units),
        nn.LeakyReLU(),
        nn.Linear(n_units, 1),
        )
      
  def forward(self, d):
    emb = self.rec(d['sensor_obs'])[0]
    _, S, C = emb.shape
    emb = emb.reshape(-1, C)[d['recurrent_last_step_mask'].bool().reshape(
      -1, S).reshape(-1)].reshape(-1, C)
    z = self.lin(emb).reshape(-1)
    
    return z

class Model():
  def __init__(self, config, df):
    self.config = config
    self.df = df

  def get_data_loader(self, mode, data, samples_per_epoch, fixed_order=False):
    sub_trajectory_keys = []
    for k in data:
      for i in range(data[k]['num_waypoints']-1):
        sub_trajectory_keys.append((k, i))
    
    dataset = ZDataset(
      sub_trajectory_keys,
      data,
      samples_per_epoch,
      self.config['sensor_cols'],
      self.config['ser_limit'],
      fixed_order,
      )

    loader = torch.utils.data.DataLoader(
      dataset, batch_size=self.config['batch_size'],
      num_workers=self.config['n_workers'], shuffle=(mode != 'test'),
      pin_memory=True)
    
    return loader, sub_trajectory_keys
  
  def fit(
      self, mode, train, valid, predict, model_folder, override_model_ext):
    train_model = override_model_ext is None
    
    self.nn = ZNN(
      ser_limit=self.config['ser_limit'],
      n_units=self.config['n_units'],
      n_inputs=len(self.config['sensor_cols']),
      num_recurrent_layers=self.config['num_recurrent_layers'],
      bidirectional_rnn=self.config['bidirectional_rnn'],
      )
    self.nn.to(self.config['device'])
    
    # Train phase
    if not train_model:
      model_str = override_model_ext
      override_model_path = model_folder / (override_model_ext + '.pt')
      override_model_state_dict = torch.load(override_model_path)
      self.nn.load_state_dict(override_model_state_dict)
    else:
      record_time = str(datetime.datetime.now())[:19]
      model_str = mode + ' - ' + record_time
      model_save_path = model_folder / (model_str + '.pt')
      
      train_loader, _ = self.get_data_loader(
        'train', train, self.config['train_samples_per_epoch'])
      valid_loader, _ = (None, None) if valid is None else (
        self.get_data_loader(
          'valid', valid, self.config['valid_samples_per_epoch']))
      
      optimizer_f = lambda par: torch.optim.Adam(par, lr=self.config['lr'])
      optimizer = optimizer_f(self.nn.parameters())
      met_hist = []
      best_valid = float('inf')
      best_train = float('inf')
      for epoch in range(self.config['n_epochs']):
        print(f"Epoch {epoch+1} of {self.config['n_epochs']}")
        start_time = time.time()
        self.nn.train()
        avg_train_loss = 0
        all_preds = []
        all_y = []
        
        for batch_id, d in enumerate(train_loader):
          # print(batch_id)
          optimizer.zero_grad()
  
          for k in d.keys():
            if k in ['long_keys']:
              d[k] = d[k].to(self.config['device']).long()
            else:
              d[k] = d[k].to(self.config['device']).float()
          preds = self.nn(d)
          all_preds.append(preds.detach().cpu().numpy())
          batch_y = d['y']
          all_y.append(batch_y.cpu().numpy())
  
          loss = nn.MSELoss()(preds, batch_y)
          avg_train_loss += loss.detach().cpu() / len(train_loader)
  
          if epoch > 0:
            loss.backward()
          optimizer.step()
  
        self.nn.eval()
        train_preds = np.concatenate(all_preds)
        train_y = np.concatenate(all_y)
        train_loss = np.sqrt(((train_y-train_preds)**2).mean())
        train_mae = np.abs(train_y-train_preds).mean()
        train_elapsed = time.time() - start_time
        
        if valid is not None:
          all_preds = []
          all_y = []
          for batch_id, d in enumerate(valid_loader):
            # print(batch_id)
            for k in d.keys():
              if k in ['long_keys']:
                d[k] = d[k].to(self.config['device']).long()
              else:
                d[k] = d[k].to(self.config['device']).float()
            preds = self.nn(d).detach().cpu().numpy()
            all_preds.append(preds)
            batch_y = d['y']
            all_y.append(batch_y.cpu().numpy())
              
          val_preds = np.concatenate(all_preds)
          val_y = np.concatenate(all_y)
          
          val_loss = np.sqrt(((val_y-val_preds)**2).mean())
          val_mae = np.abs(val_y-val_preds).mean()
          met_hist.append(val_loss)
          if val_loss < best_valid:
            best_valid = val_loss
            torch.save(self.nn.state_dict(), model_save_path,
                       _use_new_zipfile_serialization=False)
          elapsed = time.time() - start_time
          # import pdb; pdb.set_trace()
          print(f"{epoch:3}: {train_loss:8.4f} {val_loss:8.4f} {val_mae:8.4f}\
 {train_elapsed:8.2f}s {elapsed:8.2f}s")
          self.metric_history = met_hist
        else:
          print(f"{epoch:3}: {train_loss:8.4f} {train_mae:8.4f}\
 {train_elapsed:8.2f}s")
          if train_loss < best_train:
            best_train = train_loss
            torch.save(self.nn.state_dict(), model_save_path,
                       _use_new_zipfile_serialization=False)
    
      del train_loader
      del valid_loader
      gc.collect()
  
    # Predict phase
    if predict is not None:
      num_predict = np.array([
        predict[1][k]['num_waypoints']-1 for k in predict[1]]).sum()
      predict_loader, predict_sub_trajectory_keys = self.get_data_loader(
        'test', predict[1], num_predict, fixed_order=True)
      
      all_preds = []
      all_y = []
      for batch_id, d in enumerate(predict_loader):
        print(batch_id)
        for k in d.keys():
          if k in ['long_keys']:
            d[k] = d[k].to(self.config['device']).long()
          else:
            d[k] = d[k].to(self.config['device']).float()
        preds = self.nn(d).detach().cpu().numpy()
        all_preds.append(preds)
        batch_y = d['y']
        all_y.append(batch_y.cpu().numpy())
        
      predict_preds = np.concatenate(all_preds)
      predict_y = np.concatenate(all_y)
      fns, sub_trajectory_ids = zip(*predict_sub_trajectory_keys)
      
      sites = []
      floors = []
      text_levels = []
      num_waypoints = []
      for fn in fns:
        target_row = np.where(self.df.fn == fn)[0][0]
        sites.append(self.df.site_id.values[target_row])
        floors.append(self.df.level.values[target_row])
        text_levels.append(self.df.text_level.values[target_row])
        if self.df['mode'].values[target_row] == 'train':
          num_waypoints.append(self.df.num_train_waypoints.values[target_row])
        else:
          num_waypoints.append(self.df.num_test_waypoints.values[target_row])
      
      predictions = pd.DataFrame({
        'site': sites,
        'floor': floors,
        'text_level': text_levels,
        'fn': fns,
        'sub_trajectory_id': sub_trajectory_ids,
        'num_waypoints': num_waypoints,
        'prediction': predict_preds,
        'actual': predict_y
        })
      predictions_path = model_folder / 'predictions' / (
        predict[0] + ' - ' + model_str + '.csv')
      predictions.to_csv(predictions_path, index=False)

config={
  'device':'cuda',
  'n_workers': 0,
  'random_state': 142,
  'batch_size': 32,
  'n_epochs': 50,
  'lr': 4e-3,
  'n_units': 32,
  'train_samples_per_epoch': int(5e3),
  'valid_samples_per_epoch': int(2e3),
  'bidirectional_rnn': False,
  'num_recurrent_layers': 3,
  'sensor_cols': sensor_cols,
  'ser_limit': 9265,
  }

gc.collect()
m = Model(config, df)
m.fit(mode, train, valid, predict, model_folder, override_model_ext)
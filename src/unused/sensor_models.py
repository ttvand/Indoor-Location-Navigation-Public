import datetime
import gc
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
import time

import utils

mode = ['train', 'test'][0]
override_model_ext = [
    None,
    # 'train - 2021-04-08 16:30:09',  # Distance model
    # 'train - 2021-04-09 09:08:49',  # Relative movement model
    'train - 2021-04-16 11:50:12',  # Distance model
    'train - 2021-04-16 13:00:38',  # Relative movement model
    'train - 2021-04-30 13:57:48',  # Absolute movement model
][0]
only_process_test_sites = True
store_predictions = True

sensor_group_cols = ['acce', 'gyro', 'ahrs']
models_group_name = [
  'sensor_distance', 'sensor_relative_movement', 'sensor_absolute_movement'][2]
data_folder = utils.get_data_folder()
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)
sensor_folder = data_folder / 'sensor_data'
save_ext = '' if only_process_test_sites else '_all_sites'
model_folder = data_folder.parent / 'Models' / models_group_name
device_id_path = data_folder / 'device_ids.pickle'

if not 'predict' in locals() or not 'loaded_mode' in locals() or (
    mode != loaded_mode) or not 'sensor_cols' in locals() or not (
      'device_map_count' in locals()):
  test_data_path = data_folder / 'test' / (
      '0a5c442ffa2664e89e18262d_reshaped.pickle')
  with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)
  all_sensor_cols = test_data['shared_time'].columns.tolist()
  sensor_cols = [
      c for c in all_sensor_cols
      if any([g in c for g in sensor_group_cols]) and not 'uncali' in c and
      (not c[:2] == 'a_')
  ]

  with open(device_id_path, 'rb') as f:
    device_ids = pickle.load(f)
  num_device_ids = np.array(list(device_ids.values())).max() + 3
  device_map_count = (device_ids, num_device_ids)
  import pdb; pdb.set_trace()

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
      ser_limit, device_map_count, distance_model, num_independent_segments,
      fixed_order):
    self.ser_limit = ser_limit
    self.distance_model = distance_model
    self.num_independent_segments = num_independent_segments
    self.fixed_order = fixed_order
    self.sample_id = 0
    n = len(sub_trajectory_keys)
    n_inputs = len(sensor_cols)

    sensor_data = np.zeros((n, ser_limit, n_inputs), dtype=np.float16)
    device_ids = np.zeros(n, dtype=np.int16)
    if num_independent_segments == 1:
      n_target_cols = 1 if distance_model else 2
      targets = -1 * np.ones((n, n_target_cols), dtype=np.float32)
      self.num_targets = n
    else:
      fns, _ = zip(*sub_trajectory_keys)
      self.num_targets = n - len(set(fns))
      targets = -1 * np.ones((self.num_targets, 2), dtype=np.float32)
      self.segment_ids = np.zeros((self.num_targets, 2), dtype=np.int32)
    num_non_padded = np.zeros(n, dtype=np.int32)

    target_id = 0
    for i in range(n):
      # print(i)
      k, r = sub_trajectory_keys[i]
      sub_traj_data = raw_data[k]['waypoint_segments'][r][sensor_cols].values
      raw_val_count = sub_traj_data.shape[0]
      num_data_rows = min(raw_val_count, ser_limit)
      sensor_data[i, :num_data_rows] = sub_traj_data[:num_data_rows]
      device_ids[i] = device_map_count[0][k]+1
      num_non_padded[i] = raw_val_count

      if raw_data[k]['relative_waypoint_distances'] is not None:
        # There is a target
        if num_independent_segments == 1:
          if distance_model:
            targets[target_id] = raw_data[k]['relative_waypoint_distances'][r]
          else:
            targets[target_id] = raw_data[k]['relative_waypoint_movement_1'][r]
          target_id += 1
        elif i < (n - 1) and fns[i] == fns[i + 1]:
          targets[target_id] = raw_data[k]['relative_waypoint_movement_2'][r]
          self.segment_ids[target_id] = [i, i + 1]
          target_id += 1
      elif (num_independent_segments == 2) and (
          i < (n - 1) and fns[i] == fns[i + 1]):
        # There is no target
        self.segment_ids[target_id] = [i, i + 1]
        target_id += 1

    self.samples_per_epoch = samples_per_epoch
    self.n = n
    self.sensor_data = torch.from_numpy(sensor_data)
    self.device_ids = torch.from_numpy(device_ids)
    self.targets = torch.from_numpy(targets)
    self.num_non_padded = num_non_padded

  def __len__(self):
    return self.samples_per_epoch

  def __getitem__(self, idx):
    if self.fixed_order:
      s_id = self.sample_id
      self.sample_id += 1
    else:
      s_id = np.random.randint(self.num_targets)

    if self.num_independent_segments == 1:
      i1 = s_id
      mask = torch.zeros(self.ser_limit, dtype=torch.bool)
      mask[:self.num_non_padded[i1]] = True
      y = self.targets[i1]
      
      return {
          'sensor_obs': self.sensor_data[i1],
          'device_id': self.device_ids[i1],
          'mask': mask,
          'y': y,
      }
    else:
      i1, i2 = self.segment_ids[s_id]
      recurrent_last_step_mask_1 = torch.zeros(
        self.ser_limit, dtype=torch.bool)
      recurrent_last_step_mask_1[self.num_non_padded[i1] - 1] = True
      recurrent_last_step_mask_2 = torch.zeros(
        self.ser_limit, dtype=torch.bool)
      recurrent_last_step_mask_2[self.num_non_padded[i2] - 1] = True
      return {
          'sensor_obs_1': self.sensor_data[i1],
          'sensor_obs_2': self.sensor_data[i2],
          'recurrent_last_step_mask_1': recurrent_last_step_mask_1,
          'recurrent_last_step_mask_2': recurrent_last_step_mask_2,
          'y': self.targets[s_id],
      }


class DistanceZNN(nn.Module):
  def __init__(self, ser_limit, n_device_ids, n_units_cnn1, n_units_rnn,
               n_units_cnn2, n_inputs, num_recurrent_layers, bidirectional_rnn,
               num_outputs, n_emb_units=8):
    super(DistanceZNN, self).__init__()
    self.ser_limit = ser_limit
    self.n_units_cnn1 = n_units_cnn1
    self.n_units_rnn = n_units_rnn
    self.n_units_cnn2 = n_units_cnn2
    self.n_inputs = n_inputs
    self.num_recurrent_layers = num_recurrent_layers
    self.bidirectional_rnn = bidirectional_rnn
    self.num_outputs = num_outputs

    self.device_emb = nn.Embedding(n_device_ids, n_emb_units)

    self.cnn1 = torch.nn.Sequential(
        torch.nn.Conv1d(
          n_inputs+n_emb_units, config['n_units_cnn1'], kernel_size=15,
          padding=7),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(n_units_cnn1, n_units_cnn1, kernel_size=7, padding=3),
        )

    n_recurrent_units = n_units_rnn // 2 if bidirectional_rnn else n_units_rnn
    self.rnn = nn.GRU(
        input_size=n_units_cnn1,
        hidden_size=n_recurrent_units,
        num_layers=num_recurrent_layers,
        dropout=0,
        batch_first=True,
        bidirectional=bidirectional_rnn)

    self.cnn2 = torch.nn.Sequential(
        torch.nn.Conv1d(n_units_rnn, n_units_cnn2, 1),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(n_units_cnn2, n_units_cnn2, kernel_size=1, padding=0),
    )

    self.last_linear = nn.Sequential(
        nn.Linear(n_units_cnn2, num_outputs),
    )

  def forward(self, d):
    dev_emb = self.device_emb(d['device_id'].long()).unsqueeze(1).repeat(
        (1, self.ser_limit, 1))
    X = self.cnn1(torch.cat((d['sensor_obs'], dev_emb), -1).transpose(
      1, 2)).transpose(1, 2)
    rec_out = self.rnn(X)[0]
    out = self.cnn2(rec_out.transpose(1, 2)).transpose(1, 2)
    out *= d['mask'].unsqueeze(-1)

    return self.last_linear(out.sum(1).float())


class RelativeMovementZNN(nn.Module):
  def __init__(self, ser_limit, n_units, n_inputs, num_recurrent_layers,
               bidirectional_rnn):
    super(RelativeMovementZNN, self).__init__()
    self.ser_limit = ser_limit
    self.n_units = n_units
    self.n_inputs = n_inputs
    self.num_recurrent_layers = num_recurrent_layers
    self.bidirectional_rnn = bidirectional_rnn

    n_recurrent_units = n_units // 2 if bidirectional_rnn else n_units
    self.rec_1 = nn.GRU(
        input_size=n_inputs,
        hidden_size=n_recurrent_units,
        num_layers=num_recurrent_layers,
        dropout=0,
        batch_first=True,
        bidirectional=bidirectional_rnn)

    self.rec_2 = nn.GRU(
        input_size=n_inputs + n_units,
        hidden_size=n_recurrent_units,
        num_layers=num_recurrent_layers,
        dropout=0,
        batch_first=True,
        bidirectional=bidirectional_rnn)

    self.lin = nn.Sequential(
        nn.Linear(n_units, n_units),
        nn.LeakyReLU(),
        nn.Linear(n_units, 2),
    )

  def forward(self, d):
    # Encode the first sub-trajectory
    first_emb = self.rec_1(d['sensor_obs_1'])[0]
    B, S, C = first_emb.shape
    first_emb = first_emb.reshape(
        -1, C)[d['recurrent_last_step_mask_1'].bool().reshape(
            -1, S).reshape(-1)].reshape(-1, C)

    # Tile and concat the embedding of the first sub-trajectory
    tiled_first_emb = first_emb.reshape(B, 1, C).repeat(1, S, 1)
    sec_rec_input = torch.cat((d['sensor_obs_2'], tiled_first_emb), -1)

    # Encode the second sub-trajectory, conditioned on the first embedding
    second_emb = self.rec_2(sec_rec_input)[0]
    second_emb = second_emb.reshape(
        -1, C)[d['recurrent_last_step_mask_2'].bool().reshape(
            -1, S).reshape(-1)].reshape(-1, C)

    z = self.lin(second_emb)

    return z


class Model():
  def __init__(self, config, df):
    self.config = config
    self.df = df
    self.distance_model = config['model_type'] == 'sensor_distance'
    self.num_independent_segments = 2 if config['model_type'] == (
      'sensor_relative_movement') else 1

  def get_data_loader(self, mode, data, samples_per_epoch, fixed_order=False):
    sub_trajectory_keys = []
    for k in data:
      for i in range(data[k]['num_waypoints'] - 1):
        sub_trajectory_keys.append((k, i))

    dataset = ZDataset(
      sub_trajectory_keys, data, samples_per_epoch, self.config['sensor_cols'],
      self.config['ser_limit'], self.config['device_map_count'],
      self.distance_model, self.num_independent_segments, fixed_order)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=self.config['batch_size'],
        num_workers=self.config['n_workers'],
        shuffle=(mode != 'test'),
        pin_memory=True)

    return loader, sub_trajectory_keys

  def fit(self, mode, train, valid, predict, model_folder, override_model_ext):
    train_model = override_model_ext is None

    if self.num_independent_segments == 1:
      num_outputs = 1 if self.distance_model else 2
      self.nn = DistanceZNN(
          ser_limit=self.config['ser_limit'],
          n_device_ids=self.config['device_map_count'][1],
          n_units_cnn1=self.config['n_units_cnn1'],
          n_units_rnn=self.config['n_units_rnn'],
          n_units_cnn2=self.config['n_units_cnn2'],
          n_inputs=len(self.config['sensor_cols']),
          num_recurrent_layers=self.config['num_recurrent_layers'],
          bidirectional_rnn=self.config['bidirectional_rnn'],
          num_outputs=num_outputs,
      )
    else:
      self.nn = RelativeMovementZNN(
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
      valid_loader, _ = (None,
                         None) if valid is None else (self.get_data_loader(
                             'valid', valid,
                             self.config['valid_samples_per_epoch']))

      optimizer_f = lambda par: torch.optim.Adam(par, lr=self.config['lr'])
      optimizer = optimizer_f(self.nn.parameters())
      if self.config.get('scheduler', None) is not None:
        self.scheduler = self.config.get('scheduler')(optimizer)
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

          if self.distance_model:
            loss = nn.MSELoss()(preds, batch_y)
          else:
            # loss = torch.dist(preds, batch_y, 2)
            loss = nn.L1Loss()(preds, batch_y)
          avg_train_loss += loss.detach().cpu() / len(train_loader)

          if epoch > 0:
            loss.backward()
          optimizer.step()

        self.nn.eval()
        train_preds = np.concatenate(all_preds)
        train_y = np.concatenate(all_y)
        error = train_y - train_preds
        if self.distance_model:
          train_loss = np.sqrt((error[:, 0]**2).mean())
          train_mae = np.abs(error[:, 0]).mean()
        else:
          distance_errors = np.sqrt((error**2).sum(1))
          train_loss = distance_errors.mean()
          train_mae = np.abs(error).mean()

        if self.config.get('scheduler', None) is not None:
          self.scheduler.step()

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
          error = val_y - val_preds
          if self.distance_model:
            val_loss = np.sqrt((error[:, 0]**2).mean())
            val_mae = np.abs(error[:, 0]).mean()
          else:
            distance_errors = np.sqrt((error**2).sum(1))
            val_loss = distance_errors.mean()
            val_mae = np.abs(error).mean()
          met_hist.append(val_loss)
          if val_loss < best_valid:
            best_valid = val_loss
            torch.save(
                self.nn.state_dict(),
                model_save_path,
                _use_new_zipfile_serialization=False)
          elapsed = time.time() - start_time
          # import pdb; pdb.set_trace()
          if self.num_independent_segments == 1:
            print(f"{epoch:3}: {train_loss:8.4f} {val_loss:8.4f}\
 {val_mae:8.4f} {train_elapsed:8.2f}s {elapsed:8.2f}s")
          else:
            print(f"{epoch:3}: {train_loss:8.4f} {val_loss:8.4f}\
 {train_elapsed:8.2f}s {elapsed:8.2f}s")
          self.metric_history = met_hist
        else:
          print(f"{epoch:3}: {train_loss:8.4f} {train_mae:8.4f}\
 {train_elapsed:8.2f}s")
          if train_loss < best_train:
            best_train = train_loss
            torch.save(
                self.nn.state_dict(),
                model_save_path,
                _use_new_zipfile_serialization=False)

      del train_loader
      del valid_loader
      gc.collect()

    # Predict phase
    if predict is not None:
      waypoint_counts = np.array(
          [predict[1][k]['num_waypoints'] for k in predict[1]])
      if self.num_independent_segments == 1:
        num_predict = (waypoint_counts - 1).sum()
      else:
        num_predict = (waypoint_counts - 2).sum()
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
      fns = np.array(fns)
      sub_trajectory_ids = np.array(sub_trajectory_ids)

      if self.num_independent_segments == 2:
        last_fn_ids = np.concatenate([fns[:-1] != fns[1:], np.array([True])])
        fns = fns[~last_fn_ids]
        sub_trajectory_ids = sub_trajectory_ids[~last_fn_ids]
        sub_trajectory_ids += 1

      sites = []
      floors = []
      text_levels = []
      num_waypoints = []
      test_floors = utils.get_test_floors(utils.get_data_folder())
      for fn in fns:
        target_row = np.where(self.df.fn == fn)[0][0]
        sites.append(self.df.site_id.values[target_row])
        floors.append(test_floors.get(fn, self.df.level.values[target_row]))
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
      })

      if self.distance_model:
        fns = predictions.fn[predictions.sub_trajectory_id == 0].values
        predictions['fraction_time_covered'] = np.concatenate(
          [predict[1][fn]['fractions_time_covered'] for fn in fns])
        predictions['prediction'] = predict_preds[:, 0]
        predictions['actual'] = predict_y[:, 0]
      else:
        predictions['prediction_x'] = predict_preds[:, 0]
        predictions['prediction_y'] = predict_preds[:, 1]
        predictions['actual_x'] = predict_y[:, 0]
        predictions['actual_y'] = predict_y[:, 1]
      predictions_path = model_folder / 'predictions' / (
          predict[0] + ' - ' + model_str + '.csv')
      # import pdb; pdb.set_trace()
      predictions.to_csv(predictions_path, index=False)


config = {
    'device': 'cuda',
    'n_workers': 0,
    'batch_size': 32,
    'n_epochs': 50,
    'lr': 4e-3,
    'scheduler': lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 20, eta_min=1e-7),
    'n_units_cnn1': 32,
    'n_units_rnn': 32,
    'n_units_cnn2': 32,
    'train_samples_per_epoch': int(5e3),
    'valid_samples_per_epoch': int(2e3),
    'bidirectional_rnn': False,
    'num_recurrent_layers': 3,
    'sensor_cols': sensor_cols,
    'ser_limit': 5713 if (
      models_group_name == 'sensor_relative_movement') else 9265,
    'model_type': models_group_name,
    'device_map_count': device_map_count,
}

gc.collect()
m = Model(config, df)
m.fit(mode, train, valid, predict, model_folder, override_model_ext)
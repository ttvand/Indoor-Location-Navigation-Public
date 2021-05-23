import numpy as np
import pandas as pd
import pickle
from src import utils
from pathlib import Path
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import os
import json
import time


def loss(data, preds):
  tgt_dist = torch.sqrt(data['x_rel']**2 + data['y_rel']**2)
  pred_dist = torch.sqrt(preds[:, 0]**2 + preds[:, 1]**2)

  return torch.sqrt(((data['x_rel'] - preds[:, 0])**2).mean()) + torch.sqrt(
      ((data['y_rel'] - preds[:, 1])**2).mean())

def metric(targets, preds):
  tgt_dist = np.sqrt((targets**2).sum(1))
  pred_dist = np.sqrt((preds**2).sum(1))
  return np.abs(targets - preds).mean(), np.sqrt(
      ((tgt_dist - pred_dist)**2).mean())

def collator(batch):
  maxlen = max([len(b['Z']) for b in batch])
  return {
      'Z':
          torch.stack([
              torch.nn.functional.pad(
                  torch.tensor(b['Z']), (0, 0, 0, maxlen - len(b['Z'])),
                  mode='constant',
                  value=0) for b in batch
          ]),
      'mask':
          torch.stack([
              torch.nn.functional.pad(
                  torch.tensor(b['mask']), (0, maxlen - len(b['mask'])),
                  mode='constant',
                  value=0) for b in batch
          ]),
      'mask_prev':
          torch.stack([
              torch.nn.functional.pad(
                  torch.tensor(b['mask_prev']),
                  (0, maxlen - len(b['mask_prev'])),
                  mode='constant',
                  value=0) for b in batch
          ]),
      'tdiff':
          torch.stack([
              torch.nn.functional.pad(
                  torch.tensor(b['tdiff']), (0, maxlen - len(b['tdiff'])),
                  mode='constant',
                  value=0) for b in batch
          ]),
      'x_rel':
          torch.tensor([b['x_rel'] for b in batch]),
      'y_rel':
          torch.tensor([b['y_rel'] for b in batch]),
  }


class ZDS(torch.utils.data.Dataset):

  def __init__(self, dataset, agg, n_wps=1, ds_type="train"):
    self.n_wps = n_wps
    self.ds_type = ds_type
    self.agg = agg
    self.sensor_cols = [
        'x_acce', 'y_acce', 'z_acce', 'x_gyro', 'y_gyro', 'z_gyro', 'x_ahrs',
        'y_ahrs', 'z_ahrs'
    ]
    res = Parallel(n_jobs=-1)(delayed(self.process_record)(fn, rec)
                              for fn, rec in tqdm(dataset.items()))
    self.l = [item for sublist in res for item in sublist[0]]
    self.desc = [item for sublist in res for item in sublist[1]]

  def __len__(self):
    return len(self.l)

  def process_record(self, fn, rec):
    l = []
    desc = []
    for i in range(len(rec['waypoint_segments'])):
      #if (self.ds_type == "train") and (i == 0):
      if (i == 0):
        continue
      seg = pd.concat(
          [rec['waypoint_segments'][i - 1], rec['waypoint_segments'][i]])
      for c in self.sensor_cols:
        seg[c] = (seg[c] - self.agg[c]['mean']) / self.agg[c]['std']
      l.append({
          'Z':
              seg[self.sensor_cols].values,
          'mask':
              np.concatenate([
                  np.zeros(len(rec['waypoint_segments'][i - 1])),
                  np.ones(len(rec['waypoint_segments'][i]))
              ]),
          'mask_prev':
              np.ones(len(rec['waypoint_segments'][i - 1])),
          'tdiff':
              seg['time'].diff().fillna(0).values,
          'x_rel':
              rec['relative_waypoint_movement_2'][i - 1, 0]
              if self.ds_type != 'test' else 0.0,
          'y_rel':
              rec['relative_waypoint_movement_2'][i - 1, 1]
              if self.ds_type != 'test' else 0.0,
      })
      desc.append({
          'site':
              rec['site'],
          'floor':
              rec['floor'],
          'fn':
              fn,
          'sub_trajectory_id':
              i,
          'num_waypoints':
              rec['num_waypoints'],
          'x_rel':
              rec['relative_waypoint_movement_2'][i - 1, 0]
              if self.ds_type != 'test' else 0.0,
          'y_rel':
              rec['relative_waypoint_movement_2'][i - 1, 1]
              if self.ds_type != 'test' else 0.0,
      })
    return l, desc

  def __getitem__(self, idx):
    return self.l[idx]

  def get_df(self):
    return pd.DataFrame(self.desc)


class ZNN(torch.nn.Module):

  def __init__(self, config):
    super(ZNN, self).__init__()
    self.config = config
    self.cnn1 = torch.nn.Sequential(
        torch.nn.Conv1d(
            config['n_inputs'],
            config['n_units_cnn1'],
            kernel_size=1,
            padding=0),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            config['n_units_cnn1'],
            config['n_units_cnn1'],
            kernel_size=5,
            padding=2),
    )

    self.rnn = torch.nn.GRU(
        config['n_units_cnn1'],
        config['n_units_rnn'],
        batch_first=True,
        bidirectional=config['rnn_bidir'],
        num_layers=config['rnn_layers'])

    self.cnn2 = torch.nn.Sequential(
        torch.nn.Conv1d((1 + config['rnn_bidir']) * config['n_units_rnn'],
                        config['n_units_cnn2'], 1),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            config['n_units_cnn2'],
            2,
            kernel_size=1,
            padding=0),
    )

  def forward(self, batch, epoch):
    if epoch < 5:
      X = self.cnn1(
          (batch['Z'] * batch['mask'].unsqueeze(-1)).float().transpose(
              1, 2)).transpose(1, 2)
      out, hidden = self.rnn(X)
      out = self.cnn2(out.transpose(1, 2)).transpose(1, 2)

      prev = out * batch['mask_prev'].unsqueeze(-1)
      prev = prev.sum(1).float()
      out = out * batch['mask'].unsqueeze(-1)
      out = out.sum(1).float()
      return out

    X = self.cnn1(batch['Z'].transpose(1, 2)).transpose(1, 2)
    out, hidden = self.rnn(X)
    out = self.cnn2(out.transpose(1, 2)).transpose(1, 2)

    prev = out * batch['mask_prev'].unsqueeze(-1)
    prev = prev.sum(1).float()
    out = out * batch['mask'].unsqueeze(-1)
    out = out.sum(1).float()

    prev = prev / (torch.sqrt((prev**2).sum(-1)) + 1e-5).unsqueeze(-1)

    cos = prev[:, 0]
    sin = prev[:, 1]

    x = out[:, 0] * cos + out[:, 0] * sin
    y = -out[:, 0] * sin + out[:, 1] * cos

    return torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], -1)


class TrainingLoop():

  def __init__(self, train_ds, valid_ds, config, base_model):
    self.config = config
    self.model = base_model
    self.loss_function = config.get('loss_function')
    self.optimizer = config.get('optimizer')(self.model.parameters())
    if config.get('scheduler', None) is not None:
      self.scheduler = config.get('scheduler')(self.optimizer)
    self.metric = config.get('metric', None)
    try:
      with open(f"{self.config.get('results_path')}/logs.json") as json_file:
        self.logger = json.load(json_file)
    except:
      self.logger = []

    if self.config.get('mixed_precision', False):
      self.scaler = torch.cuda.amp.GradScaler()

    if not os.path.exists(config.get('results_path')):
      os.makedirs(config.get('results_path'))

    self.train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.get('batch_size'),
        num_workers=config.get('num_workers', 0),
        pin_memory=True,
        collate_fn=collator,
        shuffle=True)
    if valid_ds is None:
      self.valid_loader = None
    else:
      self.valid_loader = torch.utils.data.DataLoader(
          valid_ds,
          batch_size=config.get('batch_size'),
          num_workers=config.get('num_workers', 0),
          pin_memory=True,
          collate_fn=collator,
          shuffle=False)

  def run(self):
    for epoch in range(self.config.get('n_epochs')):
      res, preds = self.run_one_epoch(epoch)
      self.logger.append(res)

      print(f" {epoch:3d}:   " +
            f"metric {res['metric'][0]:7.4f}   {res['metric'][1]:7.4f}  " +
            f"train loss {res['train_loss']:7.4f}   " +
            f"valid loss {res['valid_loss']:7.4f}   " +
            f"time {res['runtime']:5.1f}s")

      #with open(f"{self.config.get('results_path')}/logs.json", 'w') as fp:
      #  json.dump(self.logger, fp, indent=4)
    torch.save(self.model.state_dict(),
               f"{self.config.get('results_path')}/weights.pt")
    with open(f"{self.config.get('results_path')}/preds.p", 'wb') as handle:
      pickle.dump(preds, handle, protocol=4)
    return self.model, preds

  def run_one_epoch(self, epoch):

    start_time = time.time()
    acc_steps = self.config.get('gradient_accumulation_steps', 1)

    # training
    self.model.train()
    avg_loss = 0.0
    self.optimizer.zero_grad()
    for i, data in enumerate(tqdm(self.train_loader)):
      for k in data.keys():
        data[k] = data[k].to(self.config.get('device'))

      if self.config.get('mixed_precision', False):
        with torch.cuda.amp.autocast():
          preds = self.model(data, epoch)
          loss = self.loss_function(data, preds)
          avg_loss += loss / len(self.train_loader)
          loss = loss / acc_steps
          self.scaler.scale(loss).backward()

          if (i + 1) % acc_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
      else:
        preds = self.model(data, epoch)
        loss = self.loss_function(data, preds)
        avg_loss += loss / len(self.train_loader)
        loss = loss / acc_steps
        loss.backward()

        if (i + 1) % acc_steps == 0:
          self.optimizer.step()
          self.optimizer.zero_grad()
    if self.config.get('scheduler', None) is not None:
      self.scheduler.step()

    # validation
    if self.valid_loader is not None:
      all_preds = []
      all_x = []
      all_y = []
      avg_val_loss = 0.0
      self.model.eval()
      with torch.no_grad():
        for i, data in enumerate(tqdm(self.valid_loader)):
          for k in data.keys():
            data[k] = data[k].to(self.config.get('device'))

          preds = self.model(data, epoch)
          loss = self.loss_function(data, preds)
          avg_val_loss += loss / len(self.valid_loader)
          all_preds.append(preds.detach().cpu().numpy())
          all_x.append(data['x_rel'].detach().cpu().numpy())
          all_y.append(data['y_rel'].detach().cpu().numpy())

      all_preds = np.concatenate(all_preds)
      all_targets = np.hstack([
          np.concatenate(all_x).reshape(-1, 1),
          np.concatenate(all_y).reshape(-1, 1)
      ])
      metric = self.metric(all_targets, all_preds)
      return {
          'metric': metric,
          'train_loss': avg_loss.data.item(),
          'valid_loss': avg_val_loss.item(),
          'runtime': time.time() - start_time
      }, all_preds
    else:
      return {
          'metric': 0.0,
          'train_loss': avg_loss.data.item(),
          'valid_loss': 0.0,
          'runtime': time.time() - start_time
      }, []

def run(mode, fast=False):
    config = dict(
      device='cuda' if torch.cuda.is_available() else 'cpu',
      #run_type=['valid', 'test', 'cv'][2],
      run_type=mode,
      bag_size=1 if fast else 3,
      n_inputs=9,
      n_train_wps=1,
      n_epochs=1 if fast else 25,
      batch_size=64 * 4,
      num_workers=0,
      mixed_precision=True,
      n_units_cnn1=32,
      n_units_rnn=32,
      rnn_bidir=False,
      rnn_layers=2,
      n_units_cnn2=32,
      optimizer=lambda par: torch.optim.Adam(par, lr=5e-3),
      loss_function=loss,
      metric=metric,
      scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, 25, eta_min=1e-7),
      #'gradient_accumulation_steps': 1,
      train_file=["train.pickle", "train_all_sites.pickle"][0],
      #'preds_dump_freq': 1e5,
      results_path=f"data/sensor_mov2_{mode}",
    )

    if config['run_type'] == 'cv':
        data_folder = utils.get_data_folder()
        summary_path = data_folder / 'file_summary.csv'
        sensor_folder = data_folder / 'sensor_data'

        with open('data/processed/tests_stats.p', 'rb') as f:
            agg = pickle.load(f)
        with open(sensor_folder / (config['train_file']), 'rb') as f:
            train = pickle.load(f)

        kf = KFold(n_splits=5, shuffle=True, random_state=142)
        all_keys = np.array(list(train.keys()))
        fold = 0
        for train_index, valid_index in kf.split(all_keys):
            print(f"Fold {fold}")
            train_keys = all_keys[train_index]
            valid_keys = all_keys[valid_index]

            train_ds = ZDS(
                dataset={k: v for k, v in train.items() if k in train_keys},
                agg=agg,
                n_wps=config['n_train_wps'],
                ds_type="train")
            valid_ds = ZDS(
                dataset={k: v for k, v in train.items() if k in valid_keys},
                agg=agg,
                n_wps=1,
                ds_type="valid")

            bag_preds = 0
            for bag in range(config['bag_size']):
                model = ZNN(config)
                model = model.to(config['device'])
                model, preds = TrainingLoop(train_ds, valid_ds, config, model).run()
                bag_preds += preds / config['bag_size']

            res = valid_ds.get_df()
            res['x_pred'] = bag_preds[:, 0]
            res['y_pred'] = bag_preds[:, 1]
            res.to_csv(
                f"{config.get('results_path')}/preds_bag_fold_{fold}.csv", index=False)
            fold += 1
        return()
    else:
        data_folder = utils.get_data_folder()
        summary_path = data_folder / 'file_summary.csv'
        df = pd.read_csv(summary_path)
        sensor_folder = data_folder / 'sensor_data'

        with open('data/processed/tests_stats.p', 'rb') as f:
            agg = pickle.load(f)
        with open(sensor_folder / ('train.pickle'), 'rb') as f:
            train = pickle.load(f)
        with open(sensor_folder / ('valid.pickle'), 'rb') as f:
            valid = pickle.load(f)
        with open(sensor_folder / ('test.pickle'), 'rb') as f:
            test = pickle.load(f)

    if config['run_type'] == 'test':
        train_ds = torch.utils.data.ConcatDataset([
            ZDS(dataset=train,
                agg=agg,
                n_wps=config['n_train_wps'],
                ds_type="train"),
            #ZDS(dataset=valid,
            #    agg=agg,
            #    n_wps=config['n_train_wps'],
            #    ds_type="train")
        ])
        valid_ds = ZDS(dataset=test, agg=agg, n_wps=1, ds_type="test")
    else:
        train_ds = ZDS(
            dataset=train, agg=agg, n_wps=config['n_train_wps'], ds_type="train")
        valid_ds = ZDS(dataset=valid, agg=agg, n_wps=1, ds_type="valid")

    print("Data loaded")

    bag_preds = 0
    for bag in range(config['bag_size']):
        model = ZNN(config)
        model = model.to(config['device'])
        model, preds = TrainingLoop(train_ds, valid_ds, config, model).run()
        bag_preds += preds / config['bag_size']

    res = valid_ds.get_df()
    res['x_pred'] = bag_preds[:, 0]
    res['y_pred'] = bag_preds[:, 1]
    res.to_csv(f"{config.get('results_path')}/preds_bag.csv", index=False)

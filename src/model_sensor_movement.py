import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
from src import utils
import pathlib
import torch
from tqdm import tqdm
# from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import os
import json
import time


def loss(data, preds):
  return torch.sqrt(((data['x'] - preds[:, 0])**2).mean()) + torch.sqrt(
      ((data['y'] - preds[:, 1])**2).mean())

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
      'tdiff':
          torch.stack([
              torch.nn.functional.pad(
                  torch.tensor(b['tdiff']), (0, maxlen - len(b['tdiff'])),
                  mode='constant',
                  value=0) for b in batch
          ]),
      'x':
          torch.tensor([b['x'] for b in batch]),
      'y':
          torch.tensor([b['y'] for b in batch]),
      'dist':
          torch.tensor([b['dist'] for b in batch]),
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
    # res = Parallel(n_jobs=-1)(delayed(self.process_record)(fn, rec)
    #                           for fn, rec in tqdm(dataset.items()))
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
      results = [pool.apply_async(
        self.process_record, args=(fn, rec)) for fn, rec in dataset.items()]
      res = [p.get() for p in results]
    self.l = [item for sublist in res for item in sublist[0]]
    self.desc = [item for sublist in res for item in sublist[1]]

  def __len__(self):
    return len(self.l)

  def process_record(self, fn, rec):
    l = []
    desc = []
    for i in range(len(rec['waypoint_segments'])):
      for j in range(self.n_wps):
        if i + j + 1 > len(rec['waypoint_segments']):
          break
        seg = pd.concat(rec['waypoint_segments'][i:i + j + 1])
        for c in self.sensor_cols:
          seg[c] = (seg[c] - self.agg[c]['mean']) / self.agg[c]['std']
        l.append({
            'Z':
                seg[self.sensor_cols].values,
            'mask':
                np.ones(len(seg)),
            'tdiff':
                seg['time'].diff().fillna(0).values,
            'x':
                rec['waypoints'].iloc[i + j + 1]['x_waypoint'] -
                rec['waypoints'].iloc[i]['x_waypoint']
                if self.ds_type != 'test' else 0.0,
            'y':
                rec['waypoints'].iloc[i + j + 1]['y_waypoint'] -
                rec['waypoints'].iloc[i]['y_waypoint']
                if self.ds_type != 'test' else 0.0,
            'dist':
                np.sqrt((rec['waypoints'].iloc[i + j + 1]['x_waypoint'] -
                         rec['waypoints'].iloc[i]['x_waypoint'])**2 +
                        (rec['waypoints'].iloc[i + j + 1]['y_waypoint'] -
                         rec['waypoints'].iloc[i]['y_waypoint'])**2)
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
            'x':
                rec['waypoints'].iloc[i + j + 1]['x_waypoint'] -
                rec['waypoints'].iloc[i]['x_waypoint']
                if self.ds_type != 'test' else 0.0,
            'y':
                rec['waypoints'].iloc[i + j + 1]['y_waypoint'] -
                rec['waypoints'].iloc[i]['y_waypoint']
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
            kernel_size=15,
            padding=7),
        torch.nn.LeakyReLU(),
        torch.nn.Conv1d(
            config['n_units_cnn1'],
            config['n_units_cnn1'],
            kernel_size=7,
            padding=3),
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
            config['n_units_cnn2'],
            kernel_size=1,
            padding=0),
    )

    self.last_linear = torch.nn.Sequential(
        torch.nn.Linear(config['n_units_cnn2'], 2))

  def forward(self, batch):
    X = self.cnn1(batch['Z'].transpose(1, 2)).transpose(
        1, 2)  # batch * seqlen * units
    out, hidden = self.rnn(X)  # batch * seqlen * units
    out = self.cnn2(out.transpose(1, 2)).transpose(1, 2)  # batch * seqlen * 2
    out = out * batch['mask'].unsqueeze(-1)  # batch * seqlen * 2
    return self.last_linear(out.sum(1).float())


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
          preds = self.model(data)
          loss = self.loss_function(data, preds)
          avg_loss += loss / len(self.train_loader)
          loss = loss / acc_steps
          self.scaler.scale(loss).backward()

          if (i + 1) % acc_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
      else:
        preds = self.model(data)
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

          preds = self.model(data)
          loss = self.loss_function(data, preds)
          avg_val_loss += loss / len(self.valid_loader)
          all_preds.append(preds.detach().cpu().numpy())
          all_x.append(data['x'].detach().cpu().numpy())
          all_y.append(data['y'].detach().cpu().numpy())

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
    print(f"Generating sensor absolute movement predictions in mode {mode}")
    data_folder = utils.get_data_folder()
    if mode == 'cv':
      results_path = data_folder.parent / 'Models' / (
        'sensor_absolute_movement') / mode
    else:
      results_path = data_folder.parent / 'Models' / (
        'sensor_absolute_movement') / 'predictions'
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)
  
    config = dict(
      device='cuda' if torch.cuda.is_available() else 'cpu',
      #run_type=['valid', 'test', 'cv'][2],
      run_type=mode,
      bag_size=1 if (fast or mode == 'cv') else 3,
      n_inputs=9,
      n_train_wps=1,
      n_epochs=1 if fast else (10 if mode == 'cv' else 20),
      batch_size=64 * 4,
      num_workers=0,
      mixed_precision=True,
      n_units_cnn1=32,
      n_units_rnn=32,
      rnn_bidir=False,
      rnn_layers=2,
      n_units_cnn2=32,
      optimizer=lambda par: torch.optim.Adam(par, lr=4e-3),
      loss_function=loss,
      metric=metric,
      scheduler=lambda optimizer: torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, 20, eta_min=1e-7),
      #'gradient_accumulation_steps': 1,
      #'preds_dump_freq': 1e5,
      train_file=["train.pickle", "train_all_sites.pickle"][0],
      results_path=results_path,
    )

    data_folder = utils.get_data_folder()
    sensor_folder = data_folder / 'sensor_data'
    processed_folder = data_folder / 'processed'

    if config['run_type'] == 'cv':
        last_fold_path = config.get('results_path') / ('preds_bag_fold_' + str(
          4) + '.csv')
        if last_fold_path.is_file():
          return
      
        with open(processed_folder / 'tests_stats.p', 'rb') as f:
            agg = pickle.load(f)
        with open(sensor_folder / (config['train_file']), 'rb') as f:
            train = pickle.load(f)

        kf = KFold(n_splits=5, shuffle=True, random_state=142)
        all_keys = np.array(list(train.keys()))
        fold = 0
        for train_index, valid_index in kf.split(all_keys):
            fold_path = config.get('results_path') / ('preds_bag_fold_' + str(
              fold) + '.csv')
          
            print(f"Fold {fold+1} of 5")
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
            res.to_csv(fold_path, index=False)
            fold += 1
        return
    else:
        results_path = config.get('results_path') / (
          "relative_movement_v3_" + config['run_type'] + ".csv")
        if results_path.is_file():
          return

        with open(processed_folder / 'tests_stats.p', 'rb') as f:
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
        print(f"Bag {bag+1} of {config['bag_size']}")
        model = ZNN(config)
        model = model.to(config['device'])
        model, preds = TrainingLoop(train_ds, valid_ds, config, model).run()
        bag_preds += preds / config['bag_size']

    res = valid_ds.get_df()
    res['x_pred'] = bag_preds[:, 0]
    res['y_pred'] = bag_preds[:, 1]
    res.to_csv(results_path, index=False)

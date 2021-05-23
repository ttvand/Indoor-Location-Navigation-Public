import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from src import utils
import time
import os
from tqdm import tqdm

def read_one_rec(rec, base_path):
    data_path = base_path / (str(Path(rec['ext_path']).with_suffix('')) + '_reshaped.pickle')
    with open(data_path, 'rb') as f:
        file_data = pickle.load(f)
        rec2 = {**rec, **file_data}
    return rec2

def read_all_recs(df):
    base_path = Path('data')
    return [read_one_rec(rec, base_path) for _,rec in tqdm(df.iterrows(), total=len(df))]

def run():
    df = pd.read_csv('data/file_summary.csv')
    start_time = time.time()
    recs = read_all_recs(
        #df[(df['test_site']) & (~df['num_train_waypoints'].isnull()) & (df['num_wifi'] > 0)]
        df[(df['test_site'])]
        )
    #print(len(recs))
    #print(f"Done in {time.time()-start_time:8.5f}s")

    agg = {}

    for col in tqdm(['x_acce', 'y_acce', 'z_acce', 'x_gyro', 'y_gyro', 'z_gyro', 'x_magn', 'y_magn', 'z_magn', 'x_ahrs', 'y_ahrs', 'z_ahrs']):
        x = np.concatenate([r['shared_time'][col] for r in recs])
        agg[col] = {"mean": x.mean(), "std": x.std()}

    for col in tqdm(['x_waypoint', 'y_waypoint']):
        x = np.concatenate([r['waypoint'][col] for r in recs if 'waypoint' in r])
        agg[col] = {"mean": x.mean(), "std": x.std()}

    x = np.concatenate([r['wifi']['rssid_wifi'] for r in recs if 'wifi' in r])
    agg['wifi'] = {'mean': x.mean(), "std": x.std(), "min": x.min(), "max":x.max()}

    for col in tqdm(['power_beac', 'rssi_beac']):
        x = np.concatenate([r['ibeacon'][col] for r in recs if 'ibeacon' in r])
        agg[col] = {'mean': x.mean(), "std": x.std(), "min": x.min(), "max":x.max()}

    agg['wifi']['max_records_per_t1'] = np.max([r['wifi'].groupby('t1_wifi')['bssid_wifi'].size().max() for r in recs if 'wifi' in r])
    agg['wifi']['max_records_per_t2'] = np.max([r['wifi'].groupby('t2_wifi')['bssid_wifi'].size().max() for r in recs if 'wifi' in r])

    agg['wifi']['max_unique_t1'] = np.max([r['wifi']['t1_wifi'].nunique() for r in recs if 'wifi' in r])
    agg['wifi']['max_unique_t2'] = np.max([r['wifi']['t2_wifi'].nunique() for r in recs if 'wifi' in r])

    agg['max_seq_len'] = np.max([len(r['shared_time']['time']) for r in recs])

    try:
        os.makedirs("data/processed")
    except FileExistsError:
        pass

    with open('data/processed/tests_stats.p', 'wb') as handle:
        pickle.dump(agg, handle, protocol=4)
    with open('data/processed/tests_ssid_wifi.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['wifi']['ssid_wifi'] for r in recs if 'wifi' in r])), handle, protocol=4)
    with open('data/processed/tests_bssid_wifi.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['wifi']['bssid_wifi'] for r in recs if 'wifi' in r])), handle, protocol=4)
    with open('data/processed/tests_id_beac_1.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['ibeacon']['id_beac_1'] for r in recs if 'ibeacon' in r])), handle, protocol=4)
    with open('data/processed/tests_id_beac_2.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['ibeacon']['id_beac_2'] for r in recs if 'ibeacon' in r])), handle, protocol=4)
    with open('data/processed/tests_id_beac_3.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['ibeacon']['id_beac_3'] for r in recs if 'ibeacon' in r])), handle, protocol=4)
    with open('data/processed/tests_mac_beac.p', 'wb') as handle:
        pickle.dump(set(np.concatenate([r['ibeacon']['mac_beac'] for r in recs if 'ibeacon' in r])), handle, protocol=4)
    with open('data/processed/tests_site_id.p', 'wb') as handle:
        pickle.dump(set([r['site_id'] for r in recs]), handle, protocol=4)
    with open('data/processed/tests_level_id.p', 'wb') as handle:
        pickle.dump(set([r['site_id'] + '_' + str(r['text_level']) for r in recs]), handle, protocol=4)

    print('\nDone')



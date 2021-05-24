import pandas as pd
import utils

data_folder = utils.get_data_folder()
metadata_folder = data_folder / 'metadata'
summary_path = data_folder / 'file_summary.csv'
df = pd.read_csv(summary_path)

test_sites = set(df.site_id[df.test_site])
for s in test_sites:
  floors = set(df.text_level[(df.site_id == s).values
                             & (df['mode'] == 'train').values])
  meta_floors = set(path.stem for path in (metadata_folder / s).iterdir())

  assert floors == meta_floors

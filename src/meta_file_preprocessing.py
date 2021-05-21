import numpy as np
import pandas as pd
import utils

overwrite_summary = False
data_folder = utils.get_data_folder()
file_summaries = []


def get_file_summary(file_path, site_id, mode, f, sub_sub_ext, e,
                     sample_submission_counts, test_sites):
  with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()
    # First line - extract start time
    if mode == "train":
      start_time = int(lines[0].split(":")[1][:-1])
    else:
      start_time = 0

    # Second line - extract site id
    file_site_id = lines[1].split("\t")[1].split(":")[1]
    if site_id is None:
      site_id = file_site_id
    else:
      assert site_id == file_site_id

    last_line = lines[-1]
    try:
      if mode == "train":
        end_time = last_line.split(":")[1][:-1]
      else:
        end_time = last_line.split("\t")[-1]
      if end_time[-1] in ["\r", "\n"]:
        end_time = end_time[:-1]
      end_time = int(end_time)
      complete_file = True
    except:
      end_time = int(last_line.split("\t")[0])
      complete_file = False

  # Create a dictionary of the file summary
  num_acc_u = sum([1 for l in lines if "TYPE_ACCELEROMETER_UNCALIBRATED" in l])
  num_acc = sum([1 for l in lines if "TYPE_ACCELEROMETER" in l]) - num_acc_u
  num_gyr_u = sum([1 for l in lines if "TYPE_GYROSCOPE_UNCALIBRATED" in l])
  num_gyr = sum([1 for l in lines if "TYPE_GYROSCOPE" in l]) - num_gyr_u
  num_m_u = sum([1 for l in lines if "TYPE_MAGNETIC_FIELD_UNCALIBRATED" in l])
  num_m = sum([1 for l in lines if "TYPE_MAGNETIC_FIELD" in l]) - num_m_u
  num_rot_v = sum([1 for l in lines if "TYPE_ROTATION_VECTOR" in l])
  num_wifi = sum([1 for l in lines if "TYPE_WIFI" in l])
  num_bluetooth = sum([1 for l in lines if "TYPE_BEACON" in l])
  if mode == "train":
    ext_path = e.relative_to(data_folder)
    num_test_waypoints = None
    num_train_waypoints = sum([1 for l in lines if "TYPE_WAYPOINT" in l])

    floor_string = sub_sub_ext.stem
    if floor_string in utils.TEST_FLOOR_MAPPING:
      level = utils.TEST_FLOOR_MAPPING[floor_string]
    else:
      assert not site_id in test_sites
      level = utils.NON_TEST_FLOOR_MAPPING[floor_string]
    text_level = floor_string
    test_site = site_id in test_sites
  else:
    ext_path = e.relative_to(data_folder)
    num_test_waypoints = sample_submission_counts[e.stem]
    num_train_waypoints = None
    first_waypoint_x = None
    first_waypoint_y = None
    last_waypoint_x = None
    last_waypoint_y = None
    test_site = True
    level = None
    text_level = None
  duration = end_time - start_time
  num_rows = len(lines)

  if num_train_waypoints is not None:
    waypoint_lines = [l for l in lines if "TYPE_WAYPOINT" in l]
    first_waypoint_x = float(waypoint_lines[0].split("\t")[2])
    first_waypoint_y = float(waypoint_lines[0].split("\t")[3][:-1])
    try:
      last_waypoint_x = float(waypoint_lines[-1].split("\t")[2])
      last_waypoint_y = float(waypoint_lines[-1].split("\t")[3][:-1])
    except:
      print("Warning: Failed to extract last waypoint position (no biggie)")
      last_waypoint_x = None
      last_waypoint_y = None

  if num_wifi > 0:
    wifi_lines = [l for l in lines if "TYPE_WIFI" in l]
    first_last_wifi_t1 = int(wifi_lines[0].split("\t")[0])
    
    first_wifi_part = np.array([int(l.split("\t")[0]) == first_last_wifi_t1 for l in wifi_lines])
    wifi_times = np.array([int(l.split("\t")[-1][:-1]) for l in wifi_lines])
    first_last_wifi_time = wifi_times[first_wifi_part].max()
  else:
    first_last_wifi_time = None

  file_summary = {
      "site_id": file_site_id,
      "mode": mode,
      "ext_path": ext_path,
      "fn": ext_path.stem,
      "start_time": start_time,
      "end_time": end_time,
      "duration": duration,
      "num_rows": num_rows,
      "num_test_waypoints": num_test_waypoints,
      "num_train_waypoints": num_train_waypoints,
      "first_waypoint_x": first_waypoint_x,
      "first_waypoint_y": first_waypoint_y,
      "last_waypoint_x": last_waypoint_x,
      "last_waypoint_y": last_waypoint_y,
      "num_accelerometer": num_acc,
      "num_accelerometer_uncalibrated": num_acc_u,
      "num_gyroscope": num_gyr,
      "num_gyroscope_uncalibrated": num_gyr_u,
      "num_magnetic_field": num_m,
      "num_magnetic_field_uncalibrated": num_m_u,
      "num_rotation_vector": num_rot_v,
      "num_wifi": num_wifi,
      "num_bluetooth": num_bluetooth,
      "level": level,
      "text_level": text_level,
      "test_site": test_site,
      "first_last_wifi_time": first_last_wifi_time,
  }

  return file_summary, site_id, complete_file


sample_submission = pd.read_csv(data_folder / "sample_submission.csv")
sample_submission_counts = {}
all_sites = []
for s in sample_submission.site_path_timestamp:
  all_sites.append(s.split("_")[0])
  file_name = s.split("_")[1]
  if file_name in sample_submission_counts:
    sample_submission_counts[file_name] += 1
  else:
    sample_submission_counts[file_name] = 1
test_sites = list(set(all_sites))

summary_path = data_folder / "file_summary.csv"
if summary_path.is_file() and not overwrite_summary:
  df = pd.read_csv(summary_path)
else:
  for mode in ["train", "test"]:
    main_folder = data_folder / mode
    main_data_folders_or_files = sorted(main_folder.iterdir())
    if mode == "train":
      # Loop over all train data and extract the site ID
      for f in main_data_folders_or_files:
        # sub_folder = main_folder / f
        sub_folder = f
        sub_sub_folders = sorted(sub_folder.iterdir())
        sub_sub_folders = [
            s for s in sub_sub_folders if not s.suffix == ".pickle"
        ]
        site_id = None
        for sub_sub_ext in sub_sub_folders:
          # sub_sub_path = sub_folder / sub_sub_ext
          sub_sub_path = sub_sub_ext
          sub_sub_files = sorted(sub_sub_path.iterdir())
          sub_sub_files = [s for s in sub_sub_files if s.suffix == ".txt"]
          for e in sub_sub_files:
            print(len(file_summaries))
            # file_path = sub_sub_path / e
            file_path = e
            file_summary, site_id, complete_file = get_file_summary(
                file_path,
                site_id,
                mode,
                f,
                sub_sub_ext,
                e,
                None,
                test_sites,
            )

            if complete_file:
              # The file train/5cd56b83e2acfd2d33b5cab0/B2/5cf72539e9d9c9000852f45b.txt seems cut short
              file_summaries.append(file_summary)
    else:
      main_data_folders_or_files = [
          s for s in main_data_folders_or_files if s.suffix == ".txt"
      ]
      for e in main_data_folders_or_files:
        site_id = None
        print(len(file_summaries))
        # file_path = main_folder / e
        file_path = e
        file_summary, site_id, _ = get_file_summary(
            file_path,
            site_id,
            mode,
            None,
            None,
            e,
            sample_submission_counts,
            test_sites,
        )

        file_summaries.append(file_summary)

  df = pd.DataFrame(file_summaries)
  df = df.astype({
      "num_test_waypoints": "Int64",
      "num_train_waypoints": "Int64",
      "level": "Int64",
      "first_last_wifi_time": "Int64",
  })

# # Potential subsequent run of the script
# if not 'text_level' in df.columns:
#   df['text_level'] = None
#   for i in range(df.shape[0]):
#     print(i)
#     if df['mode'][i] == 'train':
#       text_level = df['ext_path'][i].split('/')[2]
#       df.loc[i, 'text_level'] = text_level

df.to_csv(summary_path, index=False)

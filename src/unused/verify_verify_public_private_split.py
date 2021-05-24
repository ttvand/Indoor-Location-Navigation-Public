import numpy as np
import pandas as pd
import utils

data_folder = utils.get_data_folder()
leaderboard_type_path = data_folder / 'leaderboard_type.csv'
submission_folder = data_folder / 'submissions'
submission_path = submission_folder / 'verify_public_private.csv'

leaderboard_type = pd.read_csv(leaderboard_type_path)
submission = pd.read_csv(submission_path)

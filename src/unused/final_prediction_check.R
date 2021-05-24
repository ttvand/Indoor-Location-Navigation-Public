library(data.table)

# submit_ext = "Blend: test - 2021-05-17 09:18:04 - extended; test - 2021-05-17 09:35:26 - extended - Threshold 1.0666.csv"
# reference_submission = "submission_cost_minimization.csv"

submit_ext = c(
   "Blend: test - 2021-05-17 09:18:04 - extended; test - 2021-05-17 09:35:26 - extended; test - 2021-05-17 12:24:20 - extended - Threshold 1.0666.csv"
  ,"Blend: test - 2021-05-17 09:18:04 - extended; test - 2021-05-17 09:35:26 - extended; test - 2021-05-17 12:24:20 - extended - Threshold 1.0.csv"
)[2]
reference_submission = c(
   "submission_cost_minimization.csv"
  , "inflated - Blend: test - 2021-05-15 05:19:44 - extended; test - 2021-05-16 21:31:43 - extended - Threshold 1.0666.csv"
)[1]

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
combined_predictions_folder = file.path(
  data_folder, '..', 'Combined predictions')
submit_preds = fread(
  file.path(combined_predictions_folder, submit_ext))
test_preds_ref = fread(file.path(
  data_folder, "submissions", reference_submission))
leaderboard_types = fread(file.path(data_folder, "leaderboard_type.csv"))
leaderboard_types[, c("site", "count"):=NULL]

submit_preds[["fn"]] = sapply(
  submit_preds$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
submit_preds[["timestamp"]] = sapply(
  submit_preds$site_path_timestamp, function(x) strsplit(x, "_")[[1]][3])
submit_preds = submit_preds[order(fn, timestamp)]
test_preds_ref[["fn"]] = sapply(
  test_preds_ref$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
test_preds_ref[["timestamp"]] = sapply(
  test_preds_ref$site_path_timestamp, function(x) strsplit(x, "_")[[1]][3])
test_preds_ref = test_preds_ref[order(fn, timestamp)]

# Biggest prediction differences
table(test_preds_ref$timestamp == submit_preds$timestamp)
submit_preds = merge(submit_preds, leaderboard_types)
submit_preds$ref_floor = test_preds_ref$floor
submit_preds$ref_pred_x = test_preds_ref$x
submit_preds$ref_pred_y = test_preds_ref$y
submit_preds$ref_pred_x_diff = test_preds_ref$x - submit_preds$x
submit_preds$ref_pred_y_diff = test_preds_ref$y - submit_preds$y
submit_preds[, ref_distance := sqrt(ref_pred_x_diff**2+ref_pred_y_diff**2)]
dist_summary = submit_preds[, .(
  mean_dist=mean(ref_distance), test_type=type[1], .N), fn]
table(dist_summary$test_type, dist_summary$mean_dist >= 1e-5)

# Floor differences
floor_summary = submit_preds[,.(
  submit_floor=floor[1], reference_floor=ref_floor[1], .N), fn]
floor_summary[, floor_match:=(submit_floor == reference_floor)]

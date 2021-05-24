library(data.table)
library(plotly)

num_samples = 5**2
analysis_site = "5da1383b4db8ce0c98bc11ab"
analysis_floor = "F1"
analysis_pred_file = 'non_parametric_wifi - validation - 2021-03-22 11:20:09.csv'

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
wifi_data = fread(file.path(
  data_folder, "train", analysis_site, analysis_floor, "all_wifi.csv"))
wifi_train = wifi_data[mode == "train"]
setcolorder(wifi_train, c("t1_wifi"))
wifi_train$obs_index = 1:nrow(wifi_train)
pred_folder = file.path(data_folder, "../Models", models_group_name,
                        "predictions")
predictions = fread(file.path(pred_folder, analysis_pred_file))
floor_preds = predictions[(site == analysis_site) & (floor == analysis_floor)]
trajectory_errors = floor_preds[, .("mean_error"=mean(error), .N), fn]

bssid_counts = wifi_train[, .("count"=.N), "bssid_wifi"]
hist(bssid_counts$count, 50)
setorder(bssid_counts, -count) 

# Sample bssids and show their density over time
sample_probs = bssid_counts$count / sum(bssid_counts$count)
sampled_bssid = sample(bssid_counts$bssid_wifi, num_samples, replace=FALSE,
                       prob=sample_probs)
# sampled_bssid = bssid_counts[1:num_samples, bssid_wifi]

sample = wifi_train[bssid_wifi %in% sampled_bssid]
p <- ggplot(sample, aes(x=obs_index, fill=bssid_wifi, color=bssid_wifi)) +
  geom_density() +
  facet_wrap(~bssid_wifi, nrow=sqrt(num_samples)) +
  theme(legend.position = "none")
print(p)

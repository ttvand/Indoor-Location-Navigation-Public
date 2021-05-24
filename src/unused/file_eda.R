library(anytime)
library(data.table)
library(ggplot2)
library(plotly)

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
data = fread(file.path(data_folder, "file_summary.csv"))
holdout_ids = fread(file.path(data_folder, "holdout_ids.csv"))
wifi = fread(file.path(data_folder, "wifi_sample.csv"))

# Order the data by site and start time
data = data[with(data, order(site_id, first_last_wifi_time)), ]

# Start - end time quality check (should all be true)
table(data$duration > 0)

# Identify what fraction of training sites appears in the test data
train_sites = sort(unique(data[mode == 'train', site_id]))
test_sites = sort(unique(data[mode == 'test', site_id]))
cat("Fraction train in test sites",
    mean(data[mode == 'train'][['test_site']]))

# Inspect the duration and file length distributions for the train and test data
# The test files are typically longer than the train files!
hist(log10(data[mode == 'train' & test_site][['num_rows']]), 50)
hist(log10(data[mode == 'test'][['num_rows']]), 50)
hist(log10(data[mode == 'train' & test_site][['duration']]), 50)
hist(log10(data[mode == 'test'][['duration']]), 50)

# Identify the site imbalance, aggregated by the total number of rows
all_site_count = data[mode =='train', .(sum(num_rows), .N), site_id]
train_site_count = data[mode =='train' & test_site, .(
  sum(num_rows), "sum_targets"=sum(num_train_waypoints),
  "min_targets"=min(num_train_waypoints),
  "max_targets"=max(num_train_waypoints), .N), site_id]
train_site_count$mode = 'train'
test_site_count = data[mode == 'test' , .(
  sum(num_rows), "sum_targets"=sum(num_test_waypoints),
  "min_targets"=min(num_test_waypoints),
  "max_targets"=max(num_test_waypoints), .N), site_id]
test_site_count$mode = 'test'
combined_site_count = merge(train_site_count, test_site_count, by='site_id')
stacked_site_count = rbindlist(list(train_site_count, test_site_count))
with(combined_site_count, plot(
  V1.y, sum_targets.y, xlab="Num test rows", pch=19, xlim=c(0, 4e6)))
with(combined_site_count, plot(
  V1.x, sum_targets.x, xlab="Num train rows", pch=19, xlim=c(0, 27e6)))
with(combined_site_count, plot(
  sum_targets.x, sum_targets.y, xlab="Num train targets", pch=19))
with(combined_site_count, plot(
  log10(V1.x), log10(V1.y), xlab="Log10 num train rows",
  ylab="Log10 num test rows", pch=19))
with(combined_site_count, plot(
  V1.x, N.x, xlab="Num train rows", pch=19, xlim=c(0, 27e6)))
with(combined_site_count, plot(
  V1.y, N.y, xlab="Num test rows", pch=19, xlim=c(0, 4e6)))
with(combined_site_count, plot(
  N.x, N.y, xlab="Num train files", ylab="Num test files", pch=19,
  xlim=c(0, 1200)))
with(train_site_count, plot(
  V1, sum_targets, xlab="Num train rows", ylab="Num train waypoints", pch=19,
  xlim=c(0, 27e6), ylim=c(0, 8e3)))
ggplot(stacked_site_count, aes(fill=mode, y=log10(N), x=site_id)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
hist(log2(data[test_site == TRUE, num_train_waypoints]), 50, col="grey",
     xlim=c(0, 7))
hist(log2(data[test_site == TRUE, num_test_waypoints]), 50, col="grey",
     xlim=c(0, 7))
holdout_paths = holdout_ids[mode == 'valid'][['ext_path']]
hist(log2(data[ext_path %in% holdout_paths, num_train_waypoints]), 50,
     col="grey", xlim=c(0, 7))
hist(log2(data[!(ext_path %in% holdout_paths) & (test_site == TRUE),
               num_train_waypoints]), 50,
     col="grey", xlim=c(0, 7))
# browser()

# Verify that the following always appear in equal counts (all good) for test
# sites:
# - accelerometer (calibrated and non calibrated)
# - gyroscope (calibrated and non calibrated)
# - magnetic field (calibrated and non calibrated)
# - rotation vector
with(data, table(
  num_accelerometer == num_accelerometer_uncalibrated,
  num_accelerometer_uncalibrated == 0))
with(data, table(
  num_accelerometer == num_accelerometer_uncalibrated,
  test_site))
with(data, table(num_accelerometer == num_gyroscope))
for (d in c(
  'num_accelerometer_uncalibrated',
  'num_gyroscope',
  'num_gyroscope_uncalibrated',
  'num_magnetic_field',
  'num_magnetic_field_uncalibrated',
  'num_rotation_vector'
)){
  all_equal_test_sites = all(
    data[test_site == TRUE][["num_accelerometer"]] == data[test_site == TRUE][[
      d]])
  cat("\n", d, all_equal_test_sites)
}

# Date range of the different sites
train_date_ranges = data[mode == "train" & test_site,
     .("start_train"=anytime(min(start_time)/1000),
       "end_train"=anytime(max(start_time)/1000)), site_id]
test_date_ranges = data[mode == "test",
                        .("start_test"=anytime(min(first_last_wifi_time)/1000),
                          "end_test"=anytime(max(first_last_wifi_time)/1000)),
                        site_id]
combined_date_ranges = merge(train_date_ranges, test_date_ranges, by='site_id')

data$mode_site = paste(data$mode, data$site_id, sep="-")
ggplot(data[test_site == TRUE], aes(anytime(first_last_wifi_time/1000))) +
  geom_histogram(aes(y = ..density..), bins = 20, color = "black", fill ="grey") +
  facet_wrap(~ mode_site, nrow=2)

# Aggregate the number of ground truth labels for each floor-site pair
num_train_labels_site_floor = data[
  mode == 'train' & test_site,
  .(num_waypoints=sum(num_train_waypoints),
    row_sum=sum(num_rows), .N), .(site_id, text_level)]
num_train_labels_site = data[
  mode == 'train' & test_site,
  .(num_waypoints=sum(num_train_waypoints),
    row_sum=sum(num_rows), .N), site_id]

# All test data has wifi information
# Most training data has wifi information
# Most training data has wifi information
table(data$site_id[data$mode == 'train' & data$test_site],
      data$num_wifi[data$mode == 'train' & data$test_site] > 0)
table(data$mode[data$test_site], data$num_bluetooth[data$test_site] > 0)

for(site in test_sites){
  # browser()
  d = data[(site_id == site) & (num_wifi > 0)]
  d[["first_last_wifi_time_offset"]] = d$first_last_wifi_time
  d[["first_last_wifi_time_offset"]][d$mode == 'test'] = (
    d[["first_last_wifi_time_offset"]][d$mode == 'test'] - 1000000000*0)
  num_train = sum(d$mode == 'train')
  num_test = sum(d$mode == 'test')
  title = paste0(site, " - ", num_train, " train; ", num_test, " test")
  p <- ggplot(d, aes(x=anytime(first_last_wifi_time),
                   y=anytime(first_last_wifi_time_offset),
                   col=mode)) +
    geom_point() + 
    ggtitle(title)
  print(ggplotly(p))
  
  # p <- ggplot(d, aes(x=anytime(first_last_wifi_time),
  #                    y=first_waypoint_x,
  #                    col=mode)) +
  #   geom_point() + 
  #   ggtitle(title)
  # print(ggplotly(p))
  
  e <- d[mode == 'train']
  e$trajectory_id = 1:nrow(e)
  p <- ggplot(e, aes(x=trajectory_id, y=level, col=anytime(start_time/1000))) +
    geom_point() +
    ggtitle(site)
  print(ggplotly(p))
  print(p)
}

# Plot the trajectory lengths
data$num_waypoints = data$num_train_waypoints
data[mode == "test"][["num_waypoints"]] = data[
  mode == "test", num_test_waypoints]
ggplot(data[test_site == TRUE], aes(x=num_waypoints, colour=mode)) +
  geom_histogram()
# browser()

ggplot(data[test_site == TRUE], aes(x=duration, fill=mode)) +
  geom_histogram()

num_considered_test_sites = 10
num_considered_observations = 20
test_site_train_data = data[(mode == 'train') & (
  site_id %in% test_sites[1:num_considered_test_sites]) & (
  num_wifi > 0)]
test_site_train_data$step = rowid(test_site_train_data$site_id)
num_shared_steps = min(table(test_site_train_data$site_id))
num_shared_steps = min(num_shared_steps, num_considered_observations)
test_site_train_data_long = rbindlist(
  list(test_site_train_data, test_site_train_data))
test_site_train_data_long$waypoint_x = (
  test_site_train_data_long$first_waypoint_x)
test_site_train_data_long$waypoint_y = (
  test_site_train_data_long$first_waypoint_y)
test_site_train_data_long$waypoint_x[
  (nrow(test_site_train_data)+2):(nrow(test_site_train_data_long)-0)] = head(
  test_site_train_data$last_waypoint_x, -1)
test_site_train_data_long$waypoint_y[
  (nrow(test_site_train_data)+2):(nrow(test_site_train_data_long)-0)] = head(
  test_site_train_data$last_waypoint_y, -1)
test_site_train_data_long$waypoint_mode = rep(
  c("first", "last"), each=nrow(test_site_train_data))
p <- ggplot(test_site_train_data_long[
  (step <= num_shared_steps) & (step > 1)], 
            aes(x=waypoint_x, y=waypoint_y, colour = site_id,
                shape=waypoint_mode)) +
  geom_point(show.legend = FALSE, alpha = 0.7) +
  labs(x = "X start position", y = "Y start_position") +
  transition_time(step) +
  labs(title = "Step: {frame_time}") +
  shadow_wake(wake_length = 0.1, alpha = FALSE)
p

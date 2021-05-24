library(data.table)
library(ggplot2)
library(matlib)
library(plotly)

distance_model_dist_estimates = TRUE

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
sensor_data_folder = file.path(data_folder, 'sensor_data')
distance_folder = file.path("/home/tom/Kaggle/ILN/Data files", 'dsp')
data = fread(file.path(sensor_data_folder, 'meta.csv'))
distance_predictions = fread(file.path(
  data_folder, '../Models/sensor_distance/predictions',
  'distance_valid.csv'))
rel_movement_predictions = fread(file.path(
  data_folder, '../Models/sensor_relative_movement/predictions',
  'relative_movement_v2_valid.csv'))
rel_movement_predictions[, angle_error:=atan2(y_pred, x_pred) - atan2(
  y_rel, x_rel)]
rel_movement_predictions$angle_error[
  rel_movement_predictions$angle_error < pi] = (
    rel_movement_predictions$angle_error[
      rel_movement_predictions$angle_error < pi] + 2*pi)
rel_movement_predictions$angle_error[
  rel_movement_predictions$angle_error > pi] = (
    rel_movement_predictions$angle_error[
      rel_movement_predictions$angle_error > pi] - 2*pi)
meta_segments = fread(file.path(data_folder, 'sensor_data', 'meta.csv'))

# plot(distance_predictions[sub_trajectory_id==0, pred]-0.3147,
#      distance_predictions[sub_trajectory_id==0, dist], xlim=c(0, 10),
#      ylim=c(0,10))
# abline(0, 1, col="blue")
# browser()

# browser()
mso = meta_segments[sub_trajectory_id == 0][order(plot_time)]
mso$row = 1:nrow(mso)
unique_sites = sort(unique(mso$site))
mso[, site_id:=paste0("Site ", match(site, unique_sites), "; Device ",
                      device_id)]
pmso <- ggplot(mso[(device_id <= 20) & (device_id >= 10)], aes(
  x=row, y=mean_robust_sensor_time_diff, col=factor(device_id), text=site_id)) +
  geom_point()
print(pmso)
print(ggplotly(pmso, tooltip="text"))
# browser()

device_errors = fread(file.path(data_folder, 'sensor_model_device_errors.csv'))
device_errors = device_errors[order(site, fn, sub_trajectory_id)]
device_errors$new_device_id = c(TRUE, tail(device_errors$device_id, -1) != head(
  device_errors$device_id, -1))
device_errors$first_before_last_last = c(
  FALSE, (tail(device_errors$start_time, -1) < head(
    device_errors$start_time, -1)) & (!tail(device_errors$new_device_id, -1)) &(
      !(tail(device_errors$mode, -1) == 'test')))
device_errors[, dist:=sqrt(x**2+y**2)]
if(distance_model_dist_estimates){
  distance_cv_folder = file.path(distance_folder, 'distance_cv')
  folds = lapply(0:4, function(x) {
    df = fread(file.path(distance_cv_folder, paste0(
      "preds_bag_fold_", x, ".csv")))
    df$fold = x
    
    return(df)
  })
  combined_train_folds = rbindlist(folds)
  
  valid_preds = fread(file.path(distance_folder, "distance_valid.csv"))
  valid_preds[, V1:=NULL]
  valid_preds$fold = NA
  
  all_preds = rbind(combined_train_folds, valid_preds)
  all_preds = all_preds[order(site, fn, sub_trajectory_id)]
  # browser()
  device_errors$dist_pred = NA
  device_errors$dist_pred[which(!is.na(device_errors$dist))] = all_preds$pred
} else{
  device_errors[, dist_pred:=sqrt(x_pred**2+y_pred**2)]
}

device_errors[, dist_error:=dist-dist_pred]
# device_errors[, rel_error:=error/dist]
device_errors[, rel_dist_error:=dist_error/dist_pred]
device_errors[, rel_weight:=abs(dist)/sum(abs(dist)), fn]
device_errors$section = "Middle"
device_errors$section[device_errors$sub_trajectory_id == (
  device_errors$num_waypoints-2)] = "Last"
device_errors$section[device_errors$sub_trajectory_id == 0] = "First"
middle_weight_sums = device_errors[section == "Middle", .(
  middle_weight_sum=sum(rel_weight)), fn]
device_errors = merge(device_errors, middle_weight_sums, all.x=TRUE)
device_errors = device_errors[order(device_id, first_last_wifi_time,
                                    sub_trajectory_id)]
device_errors$rel_middle_weight = 0
device_errors$rel_middle_weight[device_errors$section == "Middle"] = (
  device_errors[section == "Middle", rel_weight/middle_weight_sum])
device_errors[, angle_error:=atan2(y_pred, x_pred) - atan2(y, x)]
change_rows = which((!is.na(device_errors$angle_error)) & (
  device_errors$angle_error < pi))
device_errors$angle_error[change_rows] = (
  device_errors$angle_error[change_rows] + 2*pi)
change_rows = which((!is.na(device_errors$angle_error)) & (
  device_errors$angle_error > pi))
device_errors$angle_error[change_rows] = (
  device_errors$angle_error[change_rows] - 2*pi)
av_angle_errors = device_errors[, .(mean_angle_error=mean(angle_error), .N), device_id]

fn_dev_errors = device_errors[, .(
  site=site[1],
  floor=floor[1],
  mode=mode[1],
  train_fold=train_fold[1],
  num_waypoints=num_waypoints[1],
  total_dist=sum(dist),
  
  mean_rel_dist_error=sum(rel_dist_error*rel_weight),
  mean_abs_rel_dist_error=sum(abs(rel_dist_error)*rel_weight),
  mean_angle_error=sum(angle_error*rel_weight),
  mean_abs_angle_error=sum(abs(angle_error)*rel_weight),
  first_rel_dist_error=rel_dist_error[1],
  first_abs_rel_dist_error=abs(rel_dist_error)[1],
  first_angle_error=angle_error[1],
  first_abs_angle_error=abs(angle_error[1]),
  middle_mean_rel_dist_error=sum(rel_dist_error*rel_middle_weight),
  middle_mean_abs_rel_dist_error=sum(abs(rel_dist_error)*rel_middle_weight),
  middle_mean_angle_error=sum(angle_error*rel_middle_weight),
  middle_mean_abs_angle_error=sum(abs(angle_error)*rel_middle_weight),
  last_rel_dist_error=rel_dist_error[num_waypoints[1]-1],
  last_abs_rel_dist_error=abs(rel_dist_error)[num_waypoints[1]-1],
  last_angle_error=angle_error[num_waypoints[1]-1],
  last_abs_angle_error=abs(angle_error[num_waypoints[1]-1]),
  
  first_first_last_wifi_time=min(first_last_wifi_time),
  time=min(start_time),
  device_id=min(device_id)), fn]
fn_dev_errors$plot_time = fn_dev_errors$time
fn_dev_errors$plot_time[fn_dev_errors$mode == "test"] = (
  fn_dev_errors[mode == "test", first_first_last_wifi_time])
fn_dev_errors$row = 1:nrow(fn_dev_errors)
browser()
fwrite(fn_dev_errors, file.path(data_folder, "fn_device_errors.csv"))

target_dev_fn = c("5dccce38757dea0006080071", "5dc78aae17ffdd0006f11f24",
                  "5dcb92e07cd16800060558c0", "5dc7c15d17ffdd0006f12559",
                  "29781f56d67d950acd783d29")[4]
target_device = fn_dev_errors[fn == target_dev_fn, device_id]
target_dev_row = fn_dev_errors[fn == target_dev_fn, row]
target_time = fn_dev_errors[fn == target_dev_fn, plot_time]
# target_device=1
# browser()
p <- ggplot(fn_dev_errors[device_id == target_device], aes(
  # x=plot_time/1000, y=mean_angle_error)) +
  x=plot_time/1000, y=mean_rel_dist_error, col=mode)) +
  # x=plot_time/1000, y=mean_abs_angle_error)) +
  # x=plot_time/1000, y=mean_abs_rel_dist_error)) +
  geom_point() +
  geom_smooth(span=0.2) +
  # geom_vline(xintercept=target_time/1000, col="red") +
  ggtitle(paste0(target_dev_fn, " - ", target_device)) +
  geom_hline(yintercept=0)
print(p)
print(ggplotly(p))
browser()

plot(head(fn_dev_errors$mean_angle_error, -1),
     tail(fn_dev_errors$mean_angle_error, -1))
abline(v=0)
abline(h=0)

device_errors$sub_trajectory_id_f <- factor(device_errors$sub_trajectory_id)
p <- ggplot(device_errors[(sub_trajectory_id <= 20) & (
  num_waypoints > 5)], aes(
    x=sub_trajectory_id_f, y=angle_error, fill=sub_trajectory_id_f)) +
  geom_boxplot()
print(p)

plot(head(fn_dev_errors$mean_rel_dist_error, -1),
     tail(fn_dev_errors$mean_rel_dist_error, -1))
abline(v=0)
abline(h=0)

data[, log10_start_time_offset:=log10(start_time_offset)]
p <- ggplot(data[sub_trajectory_id == 0, ], aes(
  x=log10_start_time_offset, fill=mode)) + 
  geom_density() +
  facet_wrap(~mode, ncol=1)
print(p)

p <- ggplot(data, aes(x=duration, fill=mode)) + 
  geom_density() +
  facet_wrap(~mode, ncol=1)
print(p)

p <- ggplot(data, aes(x=mean_sensor_time_diff, fill=mode)) + 
  geom_density() +
  facet_wrap(~mode, ncol=1)
print(p)

last_data = data[sub_trajectory_id == (num_waypoints-2), ]
p <- ggplot(last_data, aes(
  x=end_time_offset, fill=mode)) + 
  geom_density() +
  facet_wrap(~mode, ncol=1)
print(p)

p <- ggplot(distance_predictions, aes(x=pred, y=dist, col=site)) + 
  geom_point(alpha=0.2) +
  facet_wrap(~site, ncol=6) +
  theme(legend.position = "none") +
  xlim(0, 30) +
  ylim(0, 30) +
  geom_abline(intercept = 0, slope = 1, col="blue", alpha=0.2)
print(p)
print(ggplotly(p))

distance_predictions[,pred_int:=factor(floor(pred))]
distance_predictions[,error:=abs(pred-dist)]
p <- ggplot(distance_predictions, aes(x=pred_int, y=error, fill=pred_int)) + 
  geom_boxplot()
print(p)
print(ggplotly(p))

x = 1:50
plot(x, 1.4*(1-1/(x**0.5+0.5)))

distance_predictions[, segment_type:="Middle"]
distance_predictions[sub_trajectory_id==0,segment_type:="First"]
distance_predictions[sub_trajectory_id==num_waypoints-2,segment_type:="Last"]
mean_distance_errors = distance_predictions[, .(
  mean_error=mean(error)), .(site, segment_type)]
p <- ggplot(mean_distance_errors, aes(
  fill=segment_type, y=mean_error, x=site)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

browser()
ms = meta_segments[mode == "valid", c(
  "fn", "sub_trajectory_id", "start_time_offset", "max_sensor_time_diff",
  "end_time_offset")]
distance_pr = merge(distance_predictions, ms)
distance_pr = distance_pr[max_sensor_time_diff < 22]
distance_pr$max_sensor_time_diff = factor(distance_pr$max_sensor_time_diff)
p <- ggplot(distance_pr[(segment_type == "First")],
            aes(x=max_sensor_time_diff, y=error)) + 
  geom_boxplot()
print(p)

p <- ggplot(distance_pr[(segment_type == "Last")],
            aes(x=end_time_offset, y=dist-pred)) + 
  geom_point()
# xlim(90, 200)
print(p)

distance_predictions[, signed_error:=dist-pred]
p <- ggplot(distance_predictions[segment_type == "First"], aes(x=signed_error))+
  geom_density() +
  geom_vline(xintercept=0, size=1.5, color="red") +
  facet_wrap(~site) 
print(p)

p <- ggplot(distance_predictions, aes(
  x=error, colour=segment_type)) + 
  geom_density()
# facet_wrap(~site, nrow=6)
print(p)

rel_movement_predictions[, x:=prediction_x-actual_x]
rel_movement_predictions[, y:=prediction_y-actual_y]
rel_movement_predictions[, type:='Error']
num_orig = nrow(rel_movement_predictions)
actual_x = rel_movement_predictions$actual_x
actual_y = rel_movement_predictions$actual_y
rel_movement_predictions = rbind(
  rel_movement_predictions, rel_movement_predictions)
rel_movement_predictions[["type"]][1:num_orig] = 'Actual'
rel_movement_predictions[["x"]][1:num_orig] = actual_x
rel_movement_predictions[["y"]][1:num_orig] = actual_y
rel_movement_predictions[, magnitude:=sqrt(x**2+y**2)]
p <- ggplot(rel_movement_predictions, aes(x=x, y=y, col=type)) + 
  geom_point(alpha=0.2) +
  facet_wrap(~site, ncol=6) +
  # theme(legend.position = "none") +
  xlim(-15, 15) +
  ylim(-15, 15)
print(p)

mean_magnitudes = rel_movement_predictions[
  , .("mean_magnitude" = mean(magnitude)), .(site, type)]
p <- ggplot(mean_magnitudes, aes(fill=type, y=mean_magnitude, x=site)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

p <- ggplot(rel_movement_predictions, aes(x=magnitude, col=type)) + 
  geom_density() +
  xlim(0, 15) +
  facet_wrap(~site, ncol=6)
print(p)

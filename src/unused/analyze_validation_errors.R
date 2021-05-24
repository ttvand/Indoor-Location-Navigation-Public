library(data.table)
library(gganimate)
library(ggplot2)
library(ggpubr)
library(rjson)

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
models_group_name = 'non_parametric_wifi'
analysis_pred_file_1 = 'non_parametric_wifi - validation - 2021-03-28 18:39:52.csv'
analysis_pred_file_2 = 'non_parametric_wifi - validation - 2021-03-28 09:12:56.csv'
animate = !TRUE
analysis_fn = c(
  '5dd5216b50e04e0006f56476',
  '5dcd020823759900063d5370',
  '5dd23c1e94e4900006126ad4',
  '5dcd5c9323759900063d590a',
  '5dada4a1aa1d300006faaa0c',
  '5dc68b6b1cda370006030947',
  '5d073bb21a69370008bc5d10')[5]

file_meta = fread(file.path(data_folder, "file_summary.csv"))
wifi_times = fread(file.path(data_folder, "train_wifi_times.csv"))
holdout_ids = fread(file.path(data_folder, "holdout_ids.csv"))
waypoints = fread(file.path(data_folder, "train_waypoints.csv"))
pred_folder = file.path(data_folder, "../Models", models_group_name,
                        "predictions")
predictions_1 = fread(file.path(pred_folder, analysis_pred_file_1))
ordered_sites = sort(unique(predictions_1$site))
site_floors = predictions_1[, paste0(site, "-", floor), .(site, floor)]$V1
predictions_1[, site_id:=match(site, ordered_sites)]
predictions_1[, floor_id:=match(paste0(site, "-", floor), site_floors)-1]
predictions_2 = fread(file.path(pred_folder, analysis_pred_file_2))
predictions_2[, site_id:=match(site, ordered_sites)]
predictions_2[, floor_id:=match(paste0(site, "-", floor), site_floors)-1]
predictions_1$pred_type = "Pred 1"
predictions_2$pred_type = "Pred 2"
av_error_fn_2 = predictions_2[, mean(error), fn]

predictions = rbind(predictions_1, predictions_2)

av_pred_error = predictions[, mean(error), pred_type]

# Inspect the min fn prediction error to inspect possibly misclassified training
# data
min_traj_err = predictions_2[
  , .("Min Error" = min(error), .N), .(fn, site, site_id, floor_id)]

# Plot the prediction error distributions
p = ggplot(predictions, aes(x=site, y=error, fill=pred_type)) +
  geom_boxplot() +
  # ylim(0, 30) +
  theme(legend.position = "none")
print(p)

avg_site_err = predictions[, .(.N, "error"=mean(error)), .(
  site, site_id, pred_type)]
avg_site_err = avg_site_err[order(site, pred_type)]
avg_floor_err = predictions[, .(.N, "error"=mean(error)), .(
  site, site_id, floor, pred_type)]
avg_floor_err = avg_floor_err[order(site, floor, pred_type)]
avg_floor_err[["floor_id"]] = rep(1:(nrow(avg_floor_err)/2), each=2)-1

fn_preds = predictions[fn == analysis_fn]
fn_preds[, x:=x_pred]
fn_preds[, y:=y_pred]
fn_preds[pred_type == "Pred 1", x:=x_actual]
fn_preds[pred_type == "Pred 1", y:=y_actual]
fn_preds[pred_type == "Pred 1", pred_type:="actual"]
fn_preds[pred_type == "Pred 2", pred_type:="prediction"]

fn_floor_str = file_meta[fn == analysis_fn, text_level]
fn_analysis_site = file_meta[fn == analysis_fn, site_id]
img_file <- file.path(
  data_folder, "metadata", fn_analysis_site, fn_floor_str, "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", fn_analysis_site, fn_floor_str, "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info
# fn_preds[, waypoint_id:=mat
fn_preds[, waypoint_id:=match(waypoint_time, sort(unique(waypoint_time)))]

times = (
  sort(unique(fn_preds$waypoint_time)) - min(fn_preds$waypoint_time))/1000
errors = round(fn_preds[pred_type == "prediction", error], 2)
p <- ggplot(fn_preds, aes(x=x, y=y, color=pred_type)) + 
  background_image(img) +
  geom_point(alpha=1) +
  coord_cartesian(xlim=c(
    train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
      train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  ggtitle(analysis_fn)

if(animate){
  p <- p +
    transition_time(waypoint_id) +
    labs(title = "Waypoint: {frame_time}, time: {times[frame_time]},\
         error: {errors[frame_time]}") +
    shadow_wake(wake_length = 0.1, alpha = FALSE)
}
print(p)

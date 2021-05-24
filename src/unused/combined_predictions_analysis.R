library(data.table)
library(ggplot2)
library(ggpubr)
library(plotly)
library(rjson)

# analysis_file = 'valid - 2021-05-05 15:16:15.csv'
# analysis_file = 'valid - 2021-05-07 14:26:18.csv'
# analysis_file = 'valid - 2021-05-11 08:15:59.csv'
analysis_file = 'valid - 2021-05-15 05:36:20.csv'
analysis_fn = c(
  '5dcd3276a4dbe7000630afa3',
  '5d60ad1f04ffc90008edf439')[1]

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
combined_predictions_folder = file.path(data_folder, '../Combined predictions')
data = fread(file.path(combined_predictions_folder, analysis_file))
file_meta = fread(file.path(data_folder, "file_summary.csv"))

full_train_summary = data[data$all_targets_on_waypoint == TRUE,.(
  selected_pen=sum(selected_total_penalty),
  nearest_pen=sum(nearest_total_penalty),
  error_sum=sum(after_optim_error)), fn]
full_train_summary[, error_ratio:=nearest_pen/selected_pen]
sum(full_train_summary$error_sum)
sum(full_train_summary[selected_pen <= nearest_pen]$error_sum)
# browser()

nrows = nrow(data)
data_long = rbind(data, data, data, data)
data_long$x = c(data$x_before_optim_pred, data$x_pred, data$x_actual,
                data$x_actual*NA)
data_long$y = c(data$y_before_optim_pred, data$y_pred, data$y_actual,
                data$y_actual*NA)
data_long$error = c(data$before_optim_error, data$after_optim_error,
                    rep(0, nrows), rep(0, nrows))
data_long$penalty = c(data$wifi_penalty, data$distance_penalty,
                      data$relative_movement_penalty,
                      data$absolute_movement_penalty)
data_long$type = rep(c("Before optim", "After optim", "Actual", "Actual"),
                     each=nrows)
data_long$penalty_type = rep(c(
  "WiFi", "Distance", "Rel movement", "Abs movement"), each=nrows)
fn_data = data[fn == analysis_fn]

mean_errors = data_long[
  , .("mean_error" = mean(error)), .(site, type)]
mean_errors_all_train_traj = data_long[data_long$all_targets_on_waypoints
  , .("mean_error" = mean(error)), .(site, type)]
p <- ggplot(mean_errors[type != "Actual"], aes(
  fill=type, y=mean_error, x=site)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

q <- ggplot(mean_errors_all_train_traj[type != "Actual"], aes(
  fill=type, y=mean_error, x=site)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(q)

mean_penalties = data_long[
  , .("mean_penalty" = mean(penalty)), .(site, penalty_type)]
mean_penalties$mean_penalty = mean_penalties$mean_penalty/rep(
  mean_penalties$mean_penalty[1:24], 4)
p <- ggplot(mean_penalties, aes(
  fill=penalty_type, y=mean_penalty, x=site)) + 
  geom_bar(position="dodge", stat="identity") + theme(
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
print(p)

fn_floor_str = file_meta[fn == analysis_fn, text_level]
fn_analysis_site = file_meta[fn == analysis_fn, site_id]
img_file <- file.path(
  data_folder, "metadata", fn_analysis_site, fn_floor_str, "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", fn_analysis_site, fn_floor_str, "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info

fn_preds = data_long[fn == analysis_fn]
p <- ggplot(fn_preds, aes(x=x, y=y, color=type, group=type)) + 
  background_image(img) +
  geom_path() +
  geom_point() +
  geom_text(aes(label=trajectory_id), nudge_y=2) +
  coord_cartesian(xlim=c(
    train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
      train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  # scale_x_continuous(limits=c(100, 200), expand = c(0,0), oob = scales::squish) +
  ggtitle(analysis_fn)
print(p)
print(ggplotly(p))
cat(fn_analysis_site, fn_floor_str)
browser()

data[, traj_len := max(trajectory_id)+1, fn]
data[, mean_after_optim_error := mean(after_optim_error), traj_len]

p <- ggplot(data[all_targets_on_waypoints == FALSE],
            aes(x=traj_len, y=mean_after_optim_error)) +
  geom_point() +
  # ylim(c(0, 17)) +
  geom_smooth()
print(ggplotly(p))

data_aow = data[all_targets_on_waypoints == TRUE]
data_aow[, mean_after_optim_error := mean(after_optim_error), traj_len]
q <- ggplot(data_aow, aes(x=traj_len, y=mean_after_optim_error)) +
  geom_point() +
  # ylim(c(0, 10)) +
  xlim(c(0, 30)) +
  geom_smooth()
print(ggplotly(q))

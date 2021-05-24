library(data.table)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)
library(RColorBrewer)

analysis_fn = c(
  'e83a1c294b5d138339149fcb',
  '7861f68134d801946e3e2795'
)[2]
waypoints_ext = '_2.0_3.0_30.0.csv'
private_pred_ext = 'test - 2021-05-13 11:55:31 - extended  - REFERENCE SUBMIT with public score 0.97.csv'

mode = 'test'
data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
file_meta = fread(file.path(data_folder, "file_summary.csv"))
test_floor_counts = fread(file.path(data_folder, 'test_floor_counts.csv'))
waypoints_folder = file.path(data_folder, 'stashed_floor_additional_waypoints')
combined_predictions_folder = file.path(
  data_folder, '..', 'Combined predictions')
data = fread(file.path(waypoints_folder, paste0(mode, waypoints_ext)))
test_predictions = fread(
  file.path(combined_predictions_folder, private_pred_ext))

train_level_counts_long = data[, .N, .(site, floor, type)]
train_level_counts = dcast(
  train_level_counts_long, site + floor ~ type, value.var = "N")

analysis_site = test_predictions[fn == analysis_fn, site][1]
analysis_floor_str = test_predictions[fn == analysis_fn, floor][1]
analysis_data = data[(floor == analysis_floor_str) & (site == analysis_site)]
analysis_data[, c("floor"):= NULL]
analysis_data$trajectory_id = NA
analysis_data$size = 0.5
analysis_floor = file_meta[(site_id == analysis_site) & (
  text_level == analysis_floor_str), level][1]

floor_waypoint_preds = test_predictions[fn == analysis_fn]
floor_waypoint_preds = floor_waypoint_preds[, c(
  "site", "trajectory_id", "x_pred", "y_pred", "leaderboard_type")]
colnames(floor_waypoint_preds) = c("site", "trajectory_id",  "x", "y", "type")
setcolorder(floor_waypoint_preds, c("site", "type", "x", "y", "trajectory_id"))
for(i in 3:nrow(floor_waypoint_preds)){
  check_x = floor_waypoint_preds$x[i]
  check_y = floor_waypoint_preds$y[i]
  prev_match = which(floor_waypoint_preds[1:(i-1), x == check_x] & (
    floor_waypoint_preds[1:(i-1), y == check_y]))
  
  if(length(prev_match)){
    new_str = paste0(
      floor_waypoint_preds$trajectory_id[prev_match[1]], "-", i-1)
    floor_waypoint_preds$trajectory_id[prev_match] = new_str
    floor_waypoint_preds$trajectory_id[i] = new_str
  }
}

floor_waypoint_preds$size = 0.55

analysis_data = rbind(analysis_data, floor_waypoint_preds)
analysis_data$type = factor(
  analysis_data$type, levels=c("generated", "train", "private", "public"))
myColorScale <- brewer.pal(4, "Set1")
names(myColorScale) <- levels(analysis_data$type)
colScale <- scale_colour_manual(name = "type",values = myColorScale)

img_file <- file.path(
  data_folder, "metadata", analysis_site, analysis_floor_str,
  "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", analysis_site, analysis_floor_str,
  "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info

as = aes(x=x, y=y, color=type, size=size)
gp = geom_point()

p <- ggplot(analysis_data, as) + 
  background_image(img) +
  gp +
  geom_text(aes(label=trajectory_id), nudge_y=2) +
  coord_cartesian(xlim=c(
    train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
      train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  ggtitle(analysis_fn) +
  theme(plot.title = element_text(size=7.5))

p <- p +
  scale_size(range = c(1, 2.5), guide="none") +
  colScale

print(ggplotly(p))

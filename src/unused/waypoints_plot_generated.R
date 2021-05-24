library(data.table)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)
library(RColorBrewer)

mode = c('valid', 'test')[2]
# waypoints_ext = '_3_3.0_30.0_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-15 05:19:44 - extended.csv' # May 15 min dist to known 3.0

# waypoints_ext = '_3_3.0_30.0_False_False_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-15 19:17:29 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints False, generate_edge_waypoints False

waypoints_ext = '_3_3.0_30.0_True_False_0.4_0.7.csv'
private_pred_ext = 'test - 2021-05-15 20:32:47 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints True, generate_edge_waypoints False

# waypoints_ext = '_3_3.0_30.0_False_True_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-15 21:06:08 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints False, generate_edge_waypoints True

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
file_meta = fread(file.path(data_folder, "file_summary.csv"))
test_floor_counts = fread(file.path(data_folder, 'test_floor_counts.csv'))
waypoints_folder = file.path(data_folder, 'stashed_floor_additional_waypoints')
combined_predictions_folder = file.path(
  data_folder, '..', 'Combined predictions')
data = fread(file.path(waypoints_folder, paste0(mode, waypoints_ext)))
plot_folder = file.path(waypoints_folder, "generated_waypoints_plots")
waypoints = fread(file.path(data_folder, "train_waypoints.csv"))
holdout_ids = fread(file.path(data_folder, "holdout_ids.csv"))
valid_fns = holdout_ids[mode == 'valid', fn]
waypoints[, valid:=fn %in% valid_fns]
test_predictions = fread(
  file.path(combined_predictions_folder, private_pred_ext))
dir.create(plot_folder, showWarnings = FALSE)

train_level_counts_long = data[, .N, .(site, floor, type)]
train_level_counts = dcast(
  train_level_counts_long, site + floor ~ type, value.var = "N")

num_floors = nrow(train_level_counts)
for(i in 1:num_floors){
  cat(paste("Floor", i, "of", num_floors))
  analysis_site = train_level_counts$site[i]
  analysis_floor_str = train_level_counts$floor[i]
  analysis_data = data[(floor == analysis_floor_str) & (site == analysis_site)]
  analysis_data[, c("floor"):= NULL]
  analysis_data$size = 0.5
  analysis_floor = file_meta[(site_id == analysis_site) & (
    text_level == analysis_floor_str), level][1]
  
  has_unseen_valid = FALSE
  if(mode == "valid"){
    floor_waypoints = waypoints[(level == analysis_floor) & (
      site_id == analysis_site)]
    analysis_counts = floor_waypoints[, .(
      "num_train"=sum(valid==FALSE), "num_valid"=sum(valid==TRUE)), .(
        site_id, x, y)]
    analysis_counts$type = "Train&Valid"
    analysis_counts[["type"]][(analysis_counts$num_train > 0) &(
      analysis_counts$num_valid == 0)] = "Train only"
    analysis_counts[["type"]][(analysis_counts$num_valid > 0) &(
      analysis_counts$num_train == 0)] = "Valid only"
    unseen_valid = analysis_counts[type=="Valid only"]
    unseen_valid[, c("num_train", "num_valid"):=NULL]
    colnames(unseen_valid) = c("site", "x", "y", "type")
    setcolorder(unseen_valid, c("site", "type", "x", "y"))
    unseen_valid$size = 0.6
    analysis_data = rbind(unseen_valid, analysis_data)
    analysis_data$type = factor(
      analysis_data$type, levels=c("generated", "train", "Valid only"))
    myColorScale <- brewer.pal(3, "Set1")
    names(myColorScale) <- levels(analysis_data$type)
    colScale <- scale_colour_manual(name = "type",values = myColorScale)
    has_unseen_valid = nrow(unseen_valid) > 0
  } else{
    floor_waypoint_preds = test_predictions[
      (numeric_floor == analysis_floor) & (site == analysis_site)]
    floor_waypoint_preds = floor_waypoint_preds[, c(
      "site", "x_pred", "y_pred", "leaderboard_type")]
    colnames(floor_waypoint_preds) = c("site", "x", "y", "type")
    setcolorder(floor_waypoint_preds, c("site", "type", "x", "y"))
    floor_waypoint_preds$size = 0.6
    analysis_data = rbind(floor_waypoint_preds, analysis_data)
    analysis_data$type = factor(
      analysis_data$type, levels=c("generated", "train", "private", "public"))
    myColorScale <- brewer.pal(4, "Set1")
    names(myColorScale) <- levels(analysis_data$type)
    colScale <- scale_colour_manual(name = "type",values = myColorScale)
    has_unseen_valid = nrow(floor_waypoint_preds) > 0
  }
  
  img_file <- file.path(
    data_folder, "metadata", analysis_site, analysis_floor_str,
    "floor_image.png")
  img_dim_file <- file.path(
    data_folder, "metadata", analysis_site, analysis_floor_str,
    "floor_info.json")
  img <- png::readPNG(img_file)
  train_floor_info = fromJSON(file=img_dim_file)$map_info
  
  num_train_grid = train_level_counts$train[i]
  num_generated_grid = train_level_counts$generated[i]
  counts_row = which((test_floor_counts$site == analysis_site) & (
    test_floor_counts$floor == analysis_floor))
  if(length(counts_row) == 1){
    num_public_waypoints = test_floor_counts$public[counts_row]
    num_private_waypoints = test_floor_counts$private[counts_row]
  } else{
    num_public_waypoints = 0
    num_private_waypoints = 0
  }
  title = paste0(
    mode, ": ", i, " of ", num_floors, " ", analysis_site, ", floor ",
    analysis_floor, "; Num train grid: ", num_train_grid,
    "; Num generated grid: ", num_generated_grid, "; public-private counts: ",
    num_public_waypoints, "-", num_private_waypoints)
  
  if(has_unseen_valid){
    as = aes(x=x, y=y, color=type, size=size)
    gp = geom_point()
  }else{
    as = aes(x=x, y=y, color=type)
    gp = geom_point(size=0.5)
  }
  
  p <- ggplot(analysis_data, as) + 
    background_image(img) +
    gp +
    coord_cartesian(xlim=c(
      train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
        train_floor_info$height*0.045, train_floor_info$height*0.955)) +
    ggtitle(paste0(title)) +
    theme(plot.title = element_text(size=7.5))
  
  if(has_unseen_valid || mode == "test"){
    p <- p +
      scale_size(range = c(0.5, 2.5), guide="none") +
      colScale
  }
  
  print(p)
  ggsave(file.path(plot_folder, paste0(title, '.png')))
}

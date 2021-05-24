library(data.table)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)
library(RColorBrewer)

skip_existing = TRUE
restrict_test_mode = c("public", "private", "both")[3]
# debug_fn = NULL
debug_fn = "36e11c56d336fd708d34ac37"
# debug_fn = c(
#   '14669681607aa88e65b0d927',
# '3ca3e65a5fc44603f0c827ba',
# '2f33560940ba465c8226c951',
# '7861f68134d801946e3e2795',
# '46c865d487fa011022507710',
# 'c7c9d2f5113d91afa2b7e94d',
# '14f45baa63b4d3a700126af6',
# '5340e354629f97c9348270c0',
# 'ea777463a91a1293fc79fc52',
# '2bc46d57dc3f3ba0480e9f08',
# '673bee9ae2df0a2d44d178ec',
# 'cad1ea3fa0502ba1bcbebd6d',
# '2edd1ed9897871186b126a68',
# 'eafbdd565c2e1e7650607e43',
# '659307d58a1bd977d58bc94f',
# '2658f7f85981f264dd9657f0',
# 'f8d84977825b44af9aa220c4',
# '2116a2a90f494f7773b550ef',
# '66744e8a5c65a748497cd7c0',
# '18ab7e85351d03d1cd691a73',
# '679845ce7fb4d5b56d6887c1',
# '5a0cd5dec226a90c3818ec03',
# 'a1a53ec2c1992d3947dd4480',
# 'e3455e53350336857caf08f0',
# '965778109ccdb3eb34af8bf0',
# '72963a8c7eb520c56f88a536',
# 'e3d5b272b2c71179e501519e',
# 'bbcf74a04032515b461cb2cc',
# '36e11c56d336fd708d34ac37',
# '7276fac9b55aa417322e20f8',
# '6883b003fd96496e062c6ee1',
# 'a4ab6c6c78b232568f975f16',
# '108d983b84def2bd48bdca36',
# '78beda79f100af67d5fd91e6',
# 'e83a1c294b5d138339149fcb',
# '3f815bfeee94086f60be0d09',
# 'ac93c263f13fb5eda552ca97',
# 'dffd0a427bc76233ef658a1d',
# 'eeb5b1d0e23c0675e095b98a',
# 'cf5c7ff181051df6e88a5787',
# '7c19987b64849e60e831c394',
# 'c06224bdefec4d102f001601',
# 'a0c05d232b01247a9ba48be0',
# '6d89334316127640cff99800',
# '3e1d46017fbfcc8136bd1e9b',
# '897457315805f1d5b209c692',
# 'c7eaa26e0dcb8d3680fd3c70')
# debug_fn = "7861f68134d801946e3e2795"

# waypoints_ext = '_3_3.0_30.0_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-15 10:15:40 - extended.csv'

# waypoints_ext = '_4_3.0_30.0_True_False_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-15 20:32:47 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints True, generate_edge_waypoints False

# waypoints_ext = '_4_2.5_30.0_True_False_0.4_0.7.csv'
# private_pred_ext = 'test - 2021-05-16 16:28:51 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints True, generate_edge_waypoints False
# waypoints_ext = '_4_1.5_30.0_True_False_0.2_0.35.csv'
# private_pred_ext = 'test - 2021-05-17 12:24:20 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints True, generate_edge_waypoints False

waypoints_ext = '_4_1.125_30.0_True_False_0.15_0.262.csv'
private_pred_ext = 'test - 2021-05-17 22:29:30 - extended.csv' # May 15 min dist to known 3.0, generate_inner_waypoints True, generate_edge_waypoints False


data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
file_meta = fread(file.path(data_folder, "file_summary.csv"))
test_floor_counts = fread(file.path(data_folder, 'test_floor_counts.csv'))
waypoints_folder = file.path(data_folder, 'stashed_floor_additional_waypoints')
combined_predictions_folder = file.path(
  data_folder, '..', 'Combined predictions')
data = fread(file.path(waypoints_folder, paste0('test', waypoints_ext)))
plot_folder = file.path(waypoints_folder, "test_prediction_plots")
test_predictions = fread(
  file.path(combined_predictions_folder, private_pred_ext))
dir.create(plot_folder, showWarnings = FALSE)

train_level_counts_long = data[, .N, .(site, floor, type)]
train_level_counts = dcast(
  train_level_counts_long, site + floor ~ type, value.var = "N")

fn_uncertainties = test_predictions[, .(
  dist_uncertainty=mean(mean_dist_uncertainty),
  abs_move_uncertainty=mean(mean_abs_move_uncertainty),
  test_mode=leaderboard_type[1]), fn]
fns = fn_uncertainties$fn
num_fn = length(fns)
for(fn_id in 1:num_fn){
  cat(paste("\nFn", fn_id, "of", num_fn))
  analysis_fn = fns[fn_id]
  fn_rows = which(test_predictions$fn == analysis_fn)
  analysis_site = test_predictions[fn_rows[1], site]
  analysis_floor_str = test_predictions[fn_rows[1], floor]
  mode = test_predictions[fn_rows[1], leaderboard_type]
  mode_id = sum(fn_uncertainties[1:fn_id, test_mode] == mode)
  num_mode_ids = nrow(fn_uncertainties[test_mode == mode])
  title = paste0(mode, " ", mode_id, " of ", num_mode_ids, ": ", analysis_fn)
  save_path = file.path(plot_folder, paste0(title, '.png'))
  
  if((restrict_test_mode != "both" && restrict_test_mode != mode) || (
    skip_existing && file.exists(save_path)) || (
      (!is.null(debug_fn)) && !(analysis_fn %in% debug_fn))) next
  
  # 1: Prediction plot
  analysis_data = data[(floor == analysis_floor_str) & (site == analysis_site)]
  analysis_data[, c("floor"):= NULL]
  analysis_data$trajectory_id = NA
  analysis_data$size = 0.5
  analysis_data$size2 = 0.5
  analysis_floor = file_meta[(site_id == analysis_site) & (
    text_level == analysis_floor_str), level][1]
  
  floor_waypoint_preds = test_predictions[fn == analysis_fn]
  floor_waypoint_preds = floor_waypoint_preds[, c(
    "site", "trajectory_id", "x_pred", "y_pred", "leaderboard_type")]
  colnames(floor_waypoint_preds) = c("site", "trajectory_id",  "x", "y", "type")
  setcolorder(
    floor_waypoint_preds, c("site", "type", "x", "y", "trajectory_id"))
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
  floor_waypoint_preds$size2 = 0.45
  
  wifi_waypoint_preds = test_predictions[fn == analysis_fn]
  wifi_waypoint_preds = wifi_waypoint_preds[, c(
    "site", "x_before_optim_pred", "y_before_optim_pred")]
  colnames(wifi_waypoint_preds) = c("site", "x", "y")
  wifi_waypoint_preds$type = "wifi"
  wifi_waypoint_preds$trajectory_id = 0:(nrow(wifi_waypoint_preds)-1)
  setcolorder(wifi_waypoint_preds,
              c("site", "type", "x", "y", "trajectory_id"))
  wifi_waypoint_preds$size = 0.55
  wifi_waypoint_preds$size2 = 0.45
  
  analysis_data = rbind(analysis_data, floor_waypoint_preds,
                        wifi_waypoint_preds)
  analysis_data$type = factor(
    analysis_data$type, levels=c(
      "generated", "train", "private", "public", "wifi"))
  myColorScale <- c(brewer.pal(4, "Set1"), "#B6875473")
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
  
  p1 <- ggplot(analysis_data, aes(x=x, y=y, color=type, size=size)) + 
    background_image(img) +
    geom_point() +
    geom_path(
      data=analysis_data[type == mode], size=0.5) +
    geom_text(aes(label=trajectory_id), nudge_y=2) +
    coord_cartesian(xlim=c(
      train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
        train_floor_info$height*0.045, train_floor_info$height*0.955)) +
    ggtitle(analysis_fn) +
    theme(plot.title = element_text(size=7.5)) +
    scale_size(range = c(0.7, 1.6), guide="none") +
    colScale
  
  x_traj_range = range(c(floor_waypoint_preds$x, wifi_waypoint_preds$x))
  y_traj_range = range(c(floor_waypoint_preds$y, wifi_waypoint_preds$y))
  if(diff(x_traj_range) > diff(y_traj_range)){
    addit_x = 4
    addit_y = 4 + (diff(x_traj_range) - diff(y_traj_range))/2
  } else{
    addit_x = 4 + (diff(y_traj_range) - diff(x_traj_range))/2
    addit_y = 4
  }
  x_traj_range = c(x_traj_range[1]-addit_x, x_traj_range[2] + addit_x)
  y_traj_range = c(y_traj_range[1]-addit_y, y_traj_range[2] + addit_y + 2)
  nudge_y = diff(y_traj_range)/30
  p2 <- ggplot(analysis_data, aes(x=x, y=y, color=type, size=size2)) + 
    geom_point() +
    geom_path(
      data=analysis_data[type == mode], arrow = arrow(ends="last"), size=0.5) +
    geom_text(aes(label=trajectory_id), nudge_y=nudge_y, size=4) +
    xlim(x_traj_range[1], x_traj_range[2]) +
    ylim(y_traj_range[1], y_traj_range[2]) +
    scale_size(range = c(2, 4.2), guide="none") +
    colScale
  
  p <- ggarrange(p1, p2, ncol=2, nrow=1)
  
  # 2: Uncertainty and penalty types plots
  dist_uncertainty = fn_uncertainties$dist_uncertainty[fn_id]
  abs_move_uncertainty = fn_uncertainties$abs_move_uncertainty[fn_id]
  pdu = ggplot(fn_uncertainties, aes(x=dist_uncertainty)) +
    geom_histogram(bins=50) +
    geom_vline(xintercept=dist_uncertainty, color="red")
  pmu = ggplot(fn_uncertainties, aes(x=abs_move_uncertainty)) +
    geom_histogram(bins=50) +
    geom_vline(xintercept=abs_move_uncertainty, color="red")
  
  penalty_summ = melt(test_predictions[fn == analysis_fn, .(
    wifi=sum(wifi_penalty), distance=sum(distance_penalty),
    rel_move=sum(relative_movement_penalty),
    abs_move=sum(absolute_movement_penalty),
    time_leak=sum(time_leak_penalty),
    wifi_dir=sum(wifi_dir_penalty),
    off_grid=sum(off_grid_penalty+off_grid_density_penalty)
  )], variable.name="penalty_type", value.name="penalty_sum")
  penp <- ggplot(penalty_summ, aes(
    x=penalty_type, y=penalty_sum, fill=penalty_type)) +
    geom_bar(stat = "identity")
  
  pu <- ggarrange(pdu, pmu, penp, ncol=3, nrow=1, widths = c(1.0, 1.0, 2.0))
  
  # 3: Sensor data shapes vs ground truth
  pred_rel_x = test_predictions[fn_rows, x_pred] - test_predictions[
    fn_rows[1], x_pred]
  pred_rel_y = test_predictions[fn_rows, y_pred] - test_predictions[
    fn_rows[1], y_pred]
  not_first_fn_rows = fn_rows[2:length(fn_rows)]
  sensor_rel_x = c(0, cumsum(
    test_predictions[not_first_fn_rows, rel_move_sensor_pred_x]))
  sensor_rel_y = c(0, cumsum(
    test_predictions[not_first_fn_rows, rel_move_sensor_pred_y]))
  num_wp = length(sensor_rel_x)
  plot_data = data.table(
    x = c(pred_rel_x, sensor_rel_x),
    y = c(pred_rel_y, sensor_rel_y),
    type = rep(c("trajectory prediction", "sensor prediction"), each=num_wp),
    label = c(floor_waypoint_preds$trajectory_id, (1:num_wp)-1)
  )
  
  x_traj_range = range(plot_data$x)
  y_traj_range = range(plot_data$y)
  if(diff(x_traj_range) > diff(y_traj_range)){
    addit_x = 4
    addit_y = 4 + (diff(x_traj_range) - diff(y_traj_range))/2
  } else{
    addit_x = 4 + (diff(y_traj_range) - diff(x_traj_range))/2
    addit_y = 4
  }
  x_traj_range = c(x_traj_range[1]-addit_x, x_traj_range[2] + addit_x)
  y_traj_range = c(y_traj_range[1]-addit_y, y_traj_range[2] + addit_y + 2)
  nudge_y = diff(y_traj_range)/40
  sensor_path_p <- ggplot(plot_data, aes(x=x, y=y, col=type)) +
    geom_point() +
    geom_path(arrow = arrow(ends="last")) +
    geom_text(aes(label=label), nudge_y=nudge_y) +
    xlim(x_traj_range[1], x_traj_range[2]) +
    ylim(y_traj_range[1], y_traj_range[2])
  
  sensor_errors = sqrt(
    (pred_rel_x-sensor_rel_x)**2 + (pred_rel_y-sensor_rel_y)**2)
  plot_data = data.table(
    waypoint = 0:(num_wp-1),
    sensor_error = sensor_errors
  )
  sensor_err_p <- ggplot(plot_data, aes(x=waypoint, y=sensor_error)) +
    geom_line()
  
  sensor_p <- ggarrange(sensor_path_p, sensor_err_p, ncol=2, nrow=1,
                        widths = c(2, 2.0))
  
  figure <- ggarrange(p, pu, sensor_p, heights = c(2, 0.7, 1.5),
                      labels = c("Map", "Uncertainty", "Sensor data"),
                      ncol=1, nrow=3)
  
  # x11()
  x11(width=21, height=12)
  # dev.new(width = 30, height = 30)
  print(figure)
  # browser()
  ggsave(save_path)
  dev.off()
}

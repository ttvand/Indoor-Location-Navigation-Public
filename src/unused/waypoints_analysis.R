library(anytime)
library(data.table)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)

test_only_mode = TRUE
include_test_in_plots = FALSE
floor_plots = FALSE

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
data = fread(file.path(data_folder, "train_waypoints.csv"))
holdout_ids = fread(file.path(data_folder, "holdout_ids.csv"))
test_floors = fread(file.path(data_folder, "test_floors.csv"))
file_meta = fread(file.path(data_folder, "file_summary.csv"))
plot_folder = file.path(data_folder, "train_valid_waypoints")
test_preds_scm = fread(file.path(
  data_folder, "submissions", "submission_cost_minimization.csv"))
# test_preds_apr11 = fread(file.path(
#   data_folder, "submissions", "inflated - test - 2021-04-11 12:30:39.csv"))
test_preds_apr11 = fread(file.path(
  data_folder, "submissions", "inflated - test - 2021-05-10 17:13:20.csv"))
test_preds_apr29 = fread(file.path(
  data_folder, "submissions", "inflated - test - 2021-04-29 19:45:51 - update additional point thresholds.csv"))
test_preds_may09 = fread(file.path(
  data_folder, "submissions", "inflated - test - 2021-05-09 21:49:30.csv"))
# test_preds = fread(file.path(
#   data_folder, "submissions", "ensembling-best-performing-notebooks.csv"))
test_preds = fread(file.path(
  data_folder, "submissions", "test - 2021-05-09 08:47:14 no_real_submit.csv"))
leaderboard_types = fread(file.path(data_folder, "leaderboard_type.csv"))
dir.create(plot_folder, showWarnings = FALSE)

public_fns = leaderboard_types[type == "public", fn]
test_preds[["site"]] = sapply(
  test_preds$site_path_timestamp, function(x) strsplit(x, "_")[[1]][1])
test_preds[["fn"]] = sapply(
  test_preds$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
test_preds_scm[["fn"]] = sapply(
  test_preds_scm$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
test_preds_apr11[["fn"]] = sapply(
  test_preds_apr11$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
test_preds_apr29[["fn"]] = sapply(
  test_preds_apr29$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
test_preds_may09[["fn"]] = sapply(
  test_preds_may09$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
valid_fns = holdout_ids[mode == 'valid', fn]
data[, valid:=fn %in% valid_fns]
data_summary = data[, .("num_waypoints"=.N, "unique_waypoint_count"=length(
  unique(x*1000+y))), .(site_id, level)]
data_summary[, repeat_ratio:=num_waypoints/unique_waypoint_count]
waypoint_counts = data[, .N, .(site_id, level, x, y)]

test_floors = merge(test_floors, file_meta[
  mode == 'test', c('fn', 'num_test_waypoints')])

test_level_counts = test_floors[, .(.N, "waypoint_sum"=sum(num_test_waypoints)),
                                .(site, level)]
train_level_counts = data[, .N, .(site_id, level)]

if(floor_plots){
  for(i in 1:nrow(train_level_counts)){
    analysis_site = train_level_counts$site_id[i]
    analysis_floor = train_level_counts$level[i]
    analysis_data = data[(level == analysis_floor) & (site_id == analysis_site)]
    analysis_counts = analysis_data[, .(
      "num_train"=sum(valid==FALSE), "num_valid"=sum(valid==TRUE)), .(x, y)]
    analysis_counts$mode = "Train&Valid"
    analysis_counts[["mode"]][(analysis_counts$num_train > 0) &(
      analysis_counts$num_valid == 0)] = "Train only"
    analysis_counts[["mode"]][(analysis_counts$num_valid > 0) &(
      analysis_counts$num_train == 0)] = "Valid only"
    analysis_counts[, c("num_train", "num_valid"):=NULL]
    
    analysis_floor_str = file_meta[fn == analysis_data$fn[1], text_level]
    img_file <- file.path(
      data_folder, "metadata", analysis_site, analysis_floor_str,
      "floor_image.png")
    img_dim_file <- file.path(
      data_folder, "metadata", analysis_site, analysis_floor_str,
      "floor_info.json")
    img <- png::readPNG(img_file)
    train_floor_info = fromJSON(file=img_dim_file)$map_info
    
    test_floor_ids = which((test_floors$site == analysis_site) & (
      test_floors$level == analysis_floor))
    num_test_files = length(test_floor_ids)
    num_test_waypoints = sum(test_floors$num_test_waypoints[test_floor_ids])
    title = paste0(analysis_site, " floor ", analysis_floor,
                   "; Test files: ", num_test_files, "; Test waypoints: ",
                   num_test_waypoints)
    
    test_points = test_preds[(site == analysis_site) & (
      floor == analysis_floor)]
    if(test_only_mode){
      if(nrow(test_points) == 0) next
      test_points[["mode"]] = "Private test"
      test_points[fn %in% public_fns, mode:="Public test"]
      test_points[, c("site_path_timestamp", "floor", "site", "fn"):=NULL]
      combined_counts = test_points
    } else{
      if(include_test_in_plots && nrow(test_points) > 0){
        test_points[["mode"]] = "Private test"
        test_points[fn %in% public_fns, mode:="Public test"]
        test_points[, c("site_path_timestamp", "floor", "site", "fn"):=NULL]
        combined_counts = rbind(analysis_counts, test_points)
      } else{
        combined_counts = analysis_counts
      }
    }
    
    p <- ggplot(combined_counts, aes(x=x, y=y, color=mode)) + 
      background_image(img) +
      geom_point() +
      coord_cartesian(xlim=c(
        train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
          train_floor_info$height*0.045, train_floor_info$height*0.955)) +
      ggtitle(paste0(title))
    
    print(p)
    test_only_ext = if (test_only_mode) 'test_viz ' else ''
    ggsave(file.path(plot_folder, paste0(test_only_ext, title, '.png')))
  }
}

test_fn = c("15d9cd90a0c1d8c0b2e060f8", "c816924dba6739d9c63021f7",
            "610f2c07b26508790d1cd355")[1]
analysis_site = file_meta[fn == test_fn, site_id]
analysis_floor = test_floors[fn == test_fn, level]
analysis_data = data[(level == analysis_floor) & (site_id == analysis_site)]
analysis_floor_str = file_meta[fn == analysis_data$fn[1], text_level]
img_file <- file.path(
  data_folder, "metadata", analysis_site, analysis_floor_str,
  "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", analysis_site, analysis_floor_str,
  "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info
analysis_counts = analysis_data[, .("num_train"=.N), .(x, y)]
analysis_counts$mode = "Train&Valid"
analysis_counts[, c("num_train"):=NULL]
analysis_counts$point_size = 0.2
scm_fn = test_preds_scm[fn == test_fn]
scm_fn$mode = "cost_minimization"
scm_fn[, c("site_path_timestamp", "floor", "fn"):=NULL]
scm_fn$point_size = 0.2
scm_fn$point_size[1] = 2
scm_fn$point_size[nrow(scm_fn)] = 0.5
apr_11_fn = test_preds_apr11[fn == test_fn]
apr_11_fn$x = apr_11_fn$x+1
apr_11_fn$mode = "april 11 (train only grid)"
apr_11_fn[, c("site_path_timestamp", "floor", "fn"):=NULL]
apr_11_fn$point_size = 0.2
apr_11_fn$point_size[1] = 2
apr_11_fn$point_size[nrow(apr_11_fn)] = 0.5
apr_29_fn = test_preds_apr29[fn == test_fn]
apr_29_fn$y = apr_29_fn$y+1
apr_29_fn$mode = "april 29 (addit train grid)"
apr_29_fn[, c("site_path_timestamp", "floor", "fn"):=NULL]
apr_29_fn$point_size = 0.2
apr_29_fn$point_size[1] = 2
apr_29_fn$point_size[nrow(apr_29_fn)] = 0.5
may_09_fn = test_preds_may09[fn == test_fn]
may_09_fn$y = may_09_fn$y+1
may_09_fn$mode = "may 09 (addit train grid)"
may_09_fn[, c("site_path_timestamp", "floor", "fn"):=NULL]
may_09_fn$point_size = 0.2
may_09_fn$point_size[1] = 2
may_09_fn$point_size[nrow(may_09_fn)] = 0.5
combined_fn = rbind(analysis_counts, scm_fn, apr_11_fn, apr_29_fn, may_09_fn)
title = paste0(analysis_site, " floor ", analysis_floor,
               "; fn: ", test_fn)
line_data = rbind(scm_fn, apr_11_fn, apr_29_fn, may_09_fn)
p <- ggplot(combined_fn, aes(x=x, y=y, color=mode)) + 
  background_image(img) +
  geom_point(aes(size=point_size)) +
  geom_path(data=line_data, aes(color=mode)) +
  scale_size_continuous(range = c(1, 6)) +
  coord_cartesian(xlim=c(
    train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
      train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  ggtitle(paste0(title))

print(p)

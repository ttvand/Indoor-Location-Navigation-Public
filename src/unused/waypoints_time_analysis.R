library(anytime)
library(data.table)
library(gganimate)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)

max_waypoint_rows = 100
animation_site_id = 8
data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
data = fread(file.path(data_folder, "train_waypoints_timed.csv"))
holdout_ids = fread(file.path(data_folder, "holdout_ids.csv"))
test_floors = fread(file.path(data_folder, "test_floors.csv"))
file_meta = fread(file.path(data_folder, "file_summary.csv"))
waypoints = fread(file.path(data_folder, "train_waypoints.csv"))

analysis_sites <- c(
  "5a0546857ecc773753327266",
  "5c3c44b80379370013e0fd2b",  # 2 - Well aligned, rare waypoints inside buildings. Relation between num waypoints and floor (?).
  "5d27075f03f801723c2e360f",  # 3 - Building misaligned, needs waypoint vertical upwards shift of 3-5 units
  "5d27096c03f801723c31e5e0",  # 4 - Well aligned, no waypoints inside buildings
  "5d27097f03f801723c320d97",  # 5 - Well aligned, no waypoints inside buildings
  "5d27099f03f801723c32511d",  # 6 - Well aligned, no waypoints inside buildings, simple building layout
  "5d2709a003f801723c3251bf",  # 7 - Well aligned, waypoints often at edges of buildings
  "5d2709b303f801723c327472",  # 8 - Well aligned, no waypoints inside buildings
  "5d2709bb03f801723c32852c",  # 9 - Well aligned, no waypoints inside buildings
  "5d2709c303f801723c3299ee",  # 10 - Well aligned, no waypoints inside buildings - this site consists of 2/3 disconnected areas on the lower floors and the mid high floors
  "5d2709d403f801723c32bd39",  # 11 - Well aligned, rare waypoints inside buildings. Circular layout.
  "5d2709e003f801723c32d896",  # 12 - Well aligned, rare waypoints inside buildings
  "5da138274db8ce0c98bbd3d2",  # 13 - Well aligned, rare waypoints inside buildings - cool shape!
  "5da1382d4db8ce0c98bbe92e",  # 14 - Well aligned, no waypoints inside buildings
  "5da138314db8ce0c98bbf3a0",  # 15 - Well aligned, occasional waypoints at edge of buildings
  "5da138364db8ce0c98bc00f1",  # 16 - Well aligned, occasional waypoints at edge of buildings
  "5da1383b4db8ce0c98bc11ab",  # 17 - Well aligned, no waypoints inside buildings
  "5da138754db8ce0c98bca82f",  # 18 - Well aligned, waypoints often at edges of buildings, sometimes inside buildings
  "5da138764db8ce0c98bcaa46",  # 19 - Well aligned, no waypoints inside buildings
  "5da1389e4db8ce0c98bd0547",  # 20 - Well aligned, no waypoints inside buildings. Some open areas seem unaccessible
  "5da138b74db8ce0c98bd4774",  # 21 - Well aligned, no waypoints inside buildings
  "5da958dd46f8266d0737457b",  # 22 - Well aligned, rare waypoints inside buildings
  "5dbc1d84c1eb61796cf7c010",  # 23 - Well aligned, no waypoints inside buildings
  "5dc8cea7659e181adb076a3f"
)
num_sites = length(analysis_sites)
all_move_data = vector(mode="list", length=num_sites)

valid_fns = holdout_ids[mode == 'valid', fn]
waypoints[, valid:=fn %in% valid_fns]

wifi = fread(file.path(data_folder, "train_wifi_times.csv"))
wifi[, valid:=fn %in% valid_fns]
wifi[, log10_last_delay:= log10((wifi_t1_times-wifi_last_t2_times)/1000)]
ggplot(wifi, aes(x=log10_last_delay, fill=site_id)) +
  geom_histogram() +
  facet_wrap(~site_id, nrow=5) +
  theme(legend.position = "none")
wifi[["wifi_time_diff"]] = c(NA, diff(wifi$wifi_t1_times)/1000)
wifi[["log2_wifi_time_diff"]] = c(NA, log2(diff(wifi$wifi_t1_times)/1000))
p <- ggplot(wifi[trajectory_index > 0], aes(x=log2_wifi_time_diff, fill=site_id)) +
  geom_histogram() +
  facet_wrap(~site_id, nrow=5) +
  theme(legend.position = "none")
print(p)

valid_wifi = wifi[valid == TRUE]
max_wifi_gaps = valid_wifi[
  trajectory_index > 0, max(wifi_time_diff), fn]

test_wifi = fread(file.path(data_folder, "test_wifi_times.csv"))
test_wifi[["wifi_time_diff"]] = c(NA, diff(test_wifi$wifi_t1_times)/1000)
test_wifi[["wifi_time_diff"]] = c(
  NA, diff(test_wifi$wifi_t1_times)/1000)
test_wifi[["log2_wifi_time_diff"]] = c(
  NA, log2(diff(test_wifi$wifi_t1_times)/1000))
test_wifi[trajectory_index == 0, wifi_time_diff:=NA]
p <- ggplot(test_wifi[trajectory_index > 0], aes(
  x=log2_wifi_time_diff, fill=site_id)) +
  geom_histogram() +
  facet_wrap(~site_id, nrow=5) +
  theme(legend.position = "none")
print(p)
browser()
p <- ggplot(test_wifi[trajectory_index > 0], aes(
  x=wifi_time_diff, fill=site_id)) +
  geom_histogram() +
  # xlim(1, 3) +
  facet_wrap(~site_id, nrow=5) +
  theme(legend.position = "none")
print(p)

for (i in 1:num_sites){
  analysis_site = analysis_sites[i]
  site = data[site_id == analysis_site]
  site$site_start_time = site$time - site$time[1]
  site$record_time_diff = c(NA, diff(site$time)/1000)
  site$move_distance = c(NA, sqrt(diff(site$x_waypoint)**2 + diff(
    site$y_waypoint)**2))
  
  move_data = site[id > 0, c("site_id", "record_time_diff", "move_distance")]
  
  all_move_data[[i]] = move_data
}

combined_move_data = rbindlist(all_move_data)
# browser()
p <- ggplot(combined_move_data, aes(
  x=record_time_diff, y=move_distance, col=site_id)) +
  geom_point(alpha=0.05) +
  facet_wrap(~site_id, nrow=5) +
  theme(legend.position = "none")

interactive_analysis_site = analysis_sites[animation_site_id]
site = data[site_id == interactive_analysis_site]
site$repeated_waypoint = c(FALSE, (abs(diff(site$x_waypoint))+abs(diff(
  site$y_waypoint))) == 0)
rep_start_time_diffs = diff(site$time)[which(site$repeated_waypoint)-1]/1000

analysis_floor = site$floor[1]
analysis_data = waypoints[
  (level == analysis_floor) & (site_id == interactive_analysis_site)]
analysis_counts = analysis_data[, .(
  "num_train"=sum(valid==FALSE), "num_valid"=sum(valid==TRUE)),.(x, y)]
analysis_counts$mode = "Both"
analysis_counts[["mode"]][(analysis_counts$num_train > 0) &(
  analysis_counts$num_valid == 0)] = "Train only"
analysis_counts[["mode"]][(analysis_counts$num_valid > 0) &(
  analysis_counts$num_train == 0)] = "Valid only"
analysis_counts[, c("num_train", "num_valid"):=NULL]

analysis_floor_str = file_meta[fn == analysis_data$fn[1], text_level]
img_file <- file.path(
  data_folder, "metadata", interactive_analysis_site, analysis_floor_str,
  "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", interactive_analysis_site, analysis_floor_str,
  "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info
analysis_counts$step_reveal = 0
analysis_counts$mode = "waypoint"

floor_data = site[floor == analysis_floor]
plot_floor = floor_data[, c("x_waypoint", "y_waypoint", "type")]
setnames(plot_floor, "x_waypoint", "x")
setnames(plot_floor, "y_waypoint", "y")
setnames(plot_floor, "type", "mode")
plot_floor = plot_floor[1:min(nrow(plot_floor), max_waypoint_rows)]
plot_floor$step_reveal = 1:nrow(plot_floor)

combined_counts = rbind(analysis_counts, plot_floor)

title = paste0(interactive_analysis_site, " - floor ", analysis_floor_str)
p <- ggplot(plot_floor, aes(x=x, y=y, color=mode)) + 
  background_image(img) +
  geom_point(alpha=1) +
  coord_cartesian(xlim=c(
    train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
      train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  # transition_time(step_reveal) +
  # shadow_mark(alpha = 0.3, size = 0.5) +
  ggtitle(title)

# p <- animate(p, fps=2)
print(p)

library(data.table)
library(ggplot2)
library(ggpubr)
library(rjson)
library(plotly)
library(RColorBrewer)
library(viridis)

signature_ext = 'sensor_signature - limit near waypoint - magnetometer - 2021-05-11 16:34:26.csv'

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
file_meta = fread(file.path(data_folder, "file_summary.csv"))
signature_folder = file.path(
  data_folder, '../Models/sensor_signature/predictions')
data = fread(file.path(signature_folder, signature_ext))
plot_folder = file.path(signature_folder, 'Floor plots')
dir.create(plot_folder, showWarnings = FALSE)

floor_train_counts = file_meta[
  (test_site == TRUE) & (mode == "train"), .N, .(site_id, level, text_level)]

num_floors = nrow(floor_train_counts)
for(i in 1:num_floors){
  cat(paste("Floor", i, "of", num_floors))
  analysis_site = floor_train_counts$site_id[i]
  analysis_floor_str = floor_train_counts$text_level[i]
  analysis_data = data[(floor == analysis_floor_str) & (site == analysis_site)]
  
  img_file <- file.path(
    data_folder, "metadata", analysis_site, analysis_floor_str,
    "floor_image.png")
  img_dim_file <- file.path(
    data_folder, "metadata", analysis_site, analysis_floor_str,
    "floor_info.json")
  img <- png::readPNG(img_file)
  train_floor_info = fromJSON(file=img_dim_file)$map_info
  
  min_lim = min(analysis_data$z_magn_mean)
  max_lim = max(analysis_data$z_magn_mean)
  p_train <- ggplot(analysis_data[mode=="train"], aes(
    x=waypoint_interp_x, y=waypoint_interp_y, z=z_magn_mean)) +
    background_image(img) +
    stat_summary_2d(fun=mean, bins=30) +
    scale_fill_viridis(limits = c(min_lim, max_lim)) +
    coord_cartesian(xlim=c(
      train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
        train_floor_info$height*0.045, train_floor_info$height*0.955))
  
  p_valid <- ggplot(analysis_data[mode=="valid"], aes(
    x=waypoint_interp_x, y=waypoint_interp_y, z=z_magn_mean)) +
    background_image(img) +
    stat_summary_2d(fun=mean, bins=30) +
    scale_fill_viridis(limits = c(min_lim, max_lim)) +
    coord_cartesian(xlim=c(
      train_floor_info$width*0.045, train_floor_info$width*0.955), ylim=c(
        train_floor_info$height*0.045, train_floor_info$height*0.955))
  
  figure <- ggarrange(p_train, p_valid, labels = c(
    "Train", "Valid"), ncol=1, nrow=2)
  print(figure)
  title = paste0(i, " of ", num_floors, " ", analysis_site, ", floor ",
                 analysis_floor_str)
  
  ggsave(file.path(plot_folder, paste0(title, '.png')))
}

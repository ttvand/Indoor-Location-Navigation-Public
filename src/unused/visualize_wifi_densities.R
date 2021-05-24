library(arrow) # https://stackoverflow.com/questions/64937524/r-arrow-error-support-for-codec-snappy-not-built and https://stackoverflow.com/questions/42115972/configuration-failed-because-libcurl-was-not-found
library(data.table)
library(ggplot2)
library(ggpubr) # https://askubuntu.com/questions/1036641/unable-to-install-r-packages-in-ubuntu-18-04
library(htmlwidgets)
library(plotly)
library(png)
library(reticulate)
library(rjson)
library(viridis)

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
data = fread(file.path(data_folder, "file_summary.csv"))
min_obs_count = 100

test_sites = sort(unique(data[mode == 'test', site_id]))
site_id = test_sites[1]
train_site_folder = file.path(
  data_folder, 'wifi_features', 'train', site_id)
floors = list.files(train_site_folder)
floor = floors[1]
floor_folder = file.path(train_site_folder, floor[1])
files = list.files(floor_folder)
wifi = rbindlist(lapply(file.path(floor_folder, files), fread))
img_file <- file.path(
  data_folder, "metadata", site_id, floor, "floor_image.png")
img_dim_file <- file.path(
  data_folder, "metadata", site_id, floor, "floor_info.json")
img <- png::readPNG(img_file)
train_floor_info = fromJSON(file=img_dim_file)$map_info

hist(table(wifi$bssid_wifi), 50)
counts = wifi[, .N, bssid_wifi]
bssid = sort(counts[N >= min_obs_count, bssid_wifi])

num_wifis = 4**2
plot_bssid_ids = sample(1:length(bssid), num_wifis)
bssid_obs = wifi[bssid_wifi %in% bssid[plot_bssid_ids]]
p <- ggplot(bssid_obs, aes(
  x=waypoint_interp_x, y=waypoint_interp_y, z=rssid_wifi)) +
  stat_summary_2d(fun = mean, bins = 50) +
  background_image(img) +
  scale_fill_viridis() +
  # geom_point() +
  facet_wrap(~ bssid_wifi, nrow=sqrt(num_wifis)) +
  labs(x = "X start position", y = "Y start_position") +
  xlim(c(train_floor_info$width*0.045, train_floor_info$width*0.955)) +
  ylim(c(train_floor_info$height*0.045, train_floor_info$height*0.955)) +
  ggtitle(paste0(site_id, ' - ', floor))
print(p)
browser()
ggplotly(p)

l <- plotly::ggplotly(p)
saveWidget(l, "interactive_wifi_strengths.html")
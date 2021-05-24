library(data.table)
library(ggplot2)
library(plotly)

mode = c("valid", "test")[2]
plot_mode = !TRUE

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
if (mode == "valid"){
  data = fread(file.path(data_folder, "valid_edge_positions_v3.csv"))
} else{
  data = fread(file.path(data_folder, "test_edge_positions_v3.csv"))
}
df = fread(file.path(data_folder, "file_summary.csv"))
df = df[test_site == TRUE]
leaderboard_type = fread(file.path(data_folder, "leaderboard_type.csv"))
test_data = data$fn[1] %in% leaderboard_type$fn
inferred_device_ids = fread(file.path(data_folder, "inferred_device_ids.csv"))

if(test_data){
  leaderboard_type[, c("site", "count"):= NULL]
  data = merge(data, leaderboard_type)
  data = data[order(site, fn)]
  pred_folder = '../Models/non_parametric_wifi/predictions/'
  
  # Analyze the leaderboard type modes for all floors
  test_path = file.path(
    data_folder, '../Combined predictions/test - 2021-05-08 15:34:56.csv')
  test = fread(test_path)
  test$site=sapply(
    test$site_path_timestamp, function(x) strsplit(x, "_")[[1]][1])
  test$fn=sapply(
    test$site_path_timestamp, function(x) strsplit(x, "_")[[1]][2])
  # browser()
  test = merge(test, leaderboard_type)
  t = test[, .N, .(site, floor, type)]
  t = t[order(site, floor, type)]
  wt = dcast(t, site + floor ~ type, value.var = "N", fill=0)
  wt = wt[order(-private)]
  # fwrite(wt, file.path(data_folder, "test_floor_counts.csv"))
  
  
  # Analyze the floor predictions for the test data
  test_floor_preds = fread(file.path(
    data_folder, paste0(pred_folder, "test_floor_pred_distances.csv")))
  test_floor_preds = merge(test_floor_preds, leaderboard_type)
  test_floor_preds[["mean_min_distance_r"]] = test_floor_preds[[
    "mean_min_distance"]] + runif(nrow(test_floor_preds))/1e9
  test_floor_preds[,order := match(mean_min_distance_r, sort(
    mean_min_distance_r)), by=fn]
  test_floor_preds[,min_mean_min_distance := min(mean_min_distance), by=fn]
  test_floor_preds[
    ,min_distance_gap := mean_min_distance-min_mean_min_distance, by=fn]
  first_test_floor_preds = test_floor_preds[order == 1]
  different_test_floor_fns = first_test_floor_preds[
    numeric_floor != reference_test_floor, fn]
  
  tfp = test_floor_preds[fn %in% different_test_floor_fns]
}
table(data$reliable_preceding, data$reliable_succeeding)
mean(data$reliable_preceding)
mean(data$reliable_succeeding)
with(data[reliable_preceding == TRUE],
     table(actual_floor == reliable_preceding_floor))
with(data[reliable_succeeding == TRUE],
     table(actual_floor == reliable_succeeding_floor))
with(data[reliable_preceding == TRUE],
     table(actual_device_id == preceding_device_id))
with(data[reliable_succeeding == TRUE],
     table(actual_device_id == succeeding_device_id))
problematic_preceding = which(data$reliable_preceding & (
  data$actual_floor != data$reliable_preceding_floor))
problematic_succeeding = which(data$reliable_succeeding & (
  data$actual_floor != data$reliable_succeeding_floor))

table(data[reliable_preceding & reliable_succeeding, (
  reliable_preceding_floor == reliable_succeeding_floor)])
problematic_ids = union(problematic_preceding, problematic_succeeding)
table(data$type[problematic_ids])
problematic_data = data[problematic_ids]

# Verify that the test floor predictions are consistent with surrounding
# train data
order_problematic_fns = c()
if (test_data){
  for (site in sort(unique(data$site))){
    fn_site = df[(site_id == site)]
    fn_site = fn_site[order(first_last_wifi_time)]
    test_ids = which(fn_site$mode == "test")
    prev_floors = fn_site$level[pmax(1, test_ids-1)]
    next_floors = fn_site$level[pmin(test_ids+1, nrow(fn_site))]
    test_fns = fn_site$fn[test_ids]
    fn_match_rows = match(test_fns, data$fn)
    test_floors = data$actual_floor[fn_match_rows]
    if(length(prev_floors) != length(next_floors)) browser()
    both_valid_ids = which(!is.na(prev_floors) & !is.na(next_floors))
    both_valid_equal_ids = both_valid_ids[
      prev_floors[both_valid_ids] == next_floors[both_valid_ids]]
    both_valid_not_equal_ids = setdiff(both_valid_ids, both_valid_equal_ids)
    if(length(both_valid_not_equal_ids) > 0){
      # The test floor should be equal either to the preceeding or succeeding 
      # floor
      test_equal_floors = (prev_floors[both_valid_not_equal_ids] == (
        test_floors[both_valid_not_equal_ids])) | (
          next_floors[both_valid_not_equal_ids] == (
            test_floors[both_valid_not_equal_ids]))
      if(!all(test_equal_floors)){
        add_problematic_fns = test_fns[
          both_valid_not_equal_ids[which(!test_equal_floors)]]
        # if('44bc288dc5e6d7b2819f88c5' %in% add_problematic_fns) browser()
        order_problematic_fns = c(order_problematic_fns, add_problematic_fns)
      }
    }
    # The test floor should be equal to the preceeding and succeeding floor
    test_equal_floors = prev_floors[both_valid_equal_ids] == (
      test_floors[both_valid_equal_ids])
    if(!all(test_equal_floors)){
      add_problematic_fns = test_fns[
        both_valid_equal_ids[which(!test_equal_floors)]]
      # if('44bc288dc5e6d7b2819f88c5' %in% add_problematic_fns) browser()
      order_problematic_fns = c(order_problematic_fns, add_problematic_fns)
    }
  }
}
order_specific_ids = setdiff(1:length(order_problematic_fns),
                             match(problematic_data$fn, order_problematic_fns))
order_specific_fns = order_problematic_fns[order_specific_ids]
order_specific_rows = match(order_specific_fns, data$fn)
table(data[order_specific_rows, reliable_preceding],
      data[order_specific_rows, reliable_succeeding])
order_specific_problematic_rows = order_specific_rows[
  data[order_specific_rows, reliable_preceding | reliable_succeeding]]
need_investigate_order_rows = data$fn[
  setdiff(order_specific_rows, order_specific_problematic_rows)]
problematic_data = rbind(
  data[match(c(
    "04029880763600640a0cf42c",
    
    # Dmitry hardest edge cases
    "2379a535c2221d54e8caf6ff",
    "e9816721ed502a5414ee6aa4",
    "6085d88383432829bafe3147",
    "310990c10f799a9b21cd4ff3",
    "c3913710fd39499173a055e7",
    "219d400e61f8e06a066572f8",
    
    # Tom hardest edge case
    "dd4cbd69218f610f27cf33c8",
    
    # Dmitry disagreements
    "862a4ac32755d252c6948424",
    "e9816721ed502a5414ee6aa4",
    
    # Test level fn corrections (all private LB)
    "5853ed01a28b1d938e25b2d7",
    "ac93c263f13fb5eda552ca97",
    "b37fa5dff7ba5b417031990d",
    
    # Fns that need to be looked into
    "c70272750cc48acef9827dcb",
    "472be94f5be907c04c932114",
    "2379a535c2221d54e8caf6ff",
    "e8bf0626e27589d807a9751d",
    "049bb468e7e166e9d6370002",
    "04029880763600640a0cf42c",
    "74e1f3f41374ba181468248d",
    
    # Probably ok but slightly unexpected data order
    "9cc4412ff73ec37e30be9d9f",
    "bfcb651b80df271b79d344dd",
    "bbe172d896a38bc15fca3062",
    "44bc288dc5e6d7b2819f88c5",
    
    # Contiguous block of misclassified floors for site
    # 5d2709d403f801723c32bd39:
    "db7b0850aed5577702f151c4",
    "2d17cc2d5c1660c51e00d505",
    "507749a1187be5b582671c62",
    
    # Hardest 10 edge cases
    "dd4cbd69218f610f27cf33c8",
    "310990c10f799a9b21cd4ff3",
    "4aaa69d3ff7e13ed5e824aa2",
    "dbc3537c9913889864ebe0c1",
    "bd6637f1fa9e0074495caaca",
    "ac83e97550ad3e7396fb618b",
    "867c78ed47ee3fc4b30b9d59",
    "bd27a637d58beacf9a51c292",
    "346bf8f7f1c62bcb4cd6e307",
    "ad40d2b657a1b847b11b013f"
    ), data$fn)],
  problematic_data, data[order_specific_rows])

problematic_fns = problematic_data$fn
if(test_data && nrow(problematic_data) > 0 && plot_mode){
  for (i in 1:length(problematic_fns)){
    fn = problematic_data$fn[i]
    site = problematic_data$site[i]
    fn_site = df[(site_id == site)]
    fn_site = fn_site[order(first_last_wifi_time)]
    fn_site$row = 1:nrow(fn_site)
    target_row = which(fn_site$fn == fn)
    fn_site[["level"]][target_row] = problematic_data$actual_floor[i]
    plot_rows = (target_row-5):(target_row+5)
    plot_rows = plot_rows[plot_rows > 0 & plot_rows <= nrow(fn_site)]
    p <- ggplot(fn_site[plot_rows], aes(x=row, y=level, col=mode)) +
      geom_point() +
      geom_vline(xintercept = target_row) +
      ggtitle(paste0("   ", fn))
    
    inspected_fn = fn
    inspected_device_id = inferred_device_ids[fn == inspected_fn, device_id]
    fn_device = inferred_device_ids[device_id == inspected_device_id]
    fn_device$row = 1:nrow(fn_device)
    target_row = which(fn_device$fn == inspected_fn)
    target_time = fn_device[row==target_row, first_last_wifi_time]
    fn_device[["floor"]][target_row] = problematic_data$actual_floor[i]
    plot_rows = (target_row-5):(target_row+5)
    plot_rows = plot_rows[plot_rows > 0 & plot_rows <= nrow(fn_device)]
    q <- ggplot(fn_device[plot_rows], aes(x=row, y=floor, col=mode)) +
      geom_point() +
      geom_vline(xintercept = target_row) +
      ggtitle(paste0("        ", inspected_device_id))
    
    r <- ggplot(fn_device[plot_rows], aes(
      x=first_last_wifi_time, y=floor, col=mode)) +
      geom_point() +
      geom_vline(xintercept = target_time)
    
    s <- ggplot(test_floor_preds[fn == inspected_fn], aes(
      x=factor(numeric_floor), y=mean_min_distance)) +
      geom_bar(stat="identity") +
      ylim(0, min(20, max(test_floor_preds[
        fn == inspected_fn, mean_min_distance])))
    
    figure <- ggarrange(p, q, r, s, labels = c(
      "Site", "Device", "Time", "Building"), ncol=2, nrow=2)
    print(figure)
    cat(fn, "\n")
    browser()
  }
}

# Analyze the distance traveled as a function of delay between trajectories for
# time leak matches
data[, preceding_distance := sqrt((first_x-preceding_x)**2 + (
  first_y-preceding_y)**2)]
p <- ggplot(data[reliable_preceding == TRUE], aes(
  x=log2(delay_preceding/1000), y=preceding_distance)) + 
  geom_point() +
  xlim(0, 8) +
  ylim(0, 30)

data[, succeeding_distance := sqrt((last_x-succeeding_x)**2 + (
  last_y-succeeding_y)**2)]
q <- ggplot(data[reliable_succeeding == TRUE], aes(
  x=log2(delay_succeeding/1000), y=succeeding_distance)) + 
  geom_point() +
  xlim(0, 8) +
  ylim(0, 30)
# View(data[(delay_preceding < 3000) & reliable_preceding & (preceding_distance > 8)])
# View(data[(delay_succeeding < 4000) & reliable_succeeding & (succeeding_distance > 8)])
print(ggplotly(p))
print(ggplotly(q))

# Analyze the time difference distribution for identical waypoints
if (mode == "valid"){
  prec_first_distances = data[, sqrt((first_x-preceding_x)**2 + (
    first_y-preceding_y)**2)]
  preceding_equal_non_reliable_ids = which((prec_first_distances == 0) & (
    !data$reliable_preceding))
  preceding_equal_reliable_ids = which((prec_first_distances == 0) & (
    data$reliable_preceding))
  hist(data[preceding_equal_reliable_ids, delay_preceding/1000], 50)
  
  succ_last_distances = data[, sqrt((last_x-succeeding_x)**2 + (
    last_y-succeeding_y)**2)]
  succeeding_equal_non_reliable_ids = which((succ_last_distances == 0) & (
    !data$reliable_succeeding))
  succeeding_equal_reliable_ids = which((succ_last_distances == 0) & (
    data$reliable_succeeding))
  hist(data[succeeding_equal_reliable_ids, delay_succeeding/1000], 50)
}

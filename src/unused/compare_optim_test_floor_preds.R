library(data.table)
library(ggplot2)
library(plotly)

data_folder = '/media/tom/cbd_drive/Kaggle/ILN/Data'
combined_predictions_folder = file.path(
  data_folder, '..', 'Combined predictions')

ref_floors_ext = 'test - 2021-05-13 11:55:31 - extended  - with public score 0.97 debated floors.csv'
override_ext = 'test - 2021-05-14 14:01:33 - extended debated floors.csv'

ref_optim = fread(file.path(combined_predictions_folder, ref_floors_ext))
override_optim = fread(file.path(combined_predictions_folder, override_ext))

ref_summ = ref_optim[, sum(selected_total_penalty), fn]
ref_summ$type = "Reference"
override_sum = override_optim[, sum(selected_total_penalty), fn]
override_sum$type = "Override"

combined = rbind(ref_summ, override_sum)

p <- ggplot(combined, aes(fill=type, y=V1, x=fn)) + 
  geom_bar(position="dodge", stat="identity") +
  theme(axis.text.x=element_text(angle=90,hjust=1,vjust=0.5))

print(ggplotly(p))

f1 = ref_optim[, .(N=.N, FL=numeric_floor[1]), fn]
f2 = override_optim[, .(N=.N, FL=numeric_floor[1]), fn]
floor_merge = merge(f1, f2, "fn")
floor_merge[, c("N.y"):=NULL]
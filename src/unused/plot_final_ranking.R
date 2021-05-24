library(data.table)
library(ggplot2)
library(stringi)

intelecy_plot = !TRUE
select_id = as.integer(intelecy_plot)+1
min_y = c(1.4, 0)[select_id]
highlight_col = c("#1E90FF", "#00DC00")[select_id]
our_team = c("Track me if you can", "Intelecy\n(Track me if you can)")[
  select_id]

teams = c(
  our_team,
  "MYRCJ",
  "Zidmie",
  "ELQMH",
  "chris",
  "Boyrin Vjacheslav",
  "IMRTE",
  "Michał Stolarczyk",
  "higepon & saito",
  "cuML",
  "Ouranos & Vicens",
  "Hugues"
)

scores = c(
  1.49717,
  2.19892,
  2.48923,
  2.68029,
  2.77024,
  2.81548,
  2.82709,
  3.07282,
  3.10582,
  3.12258,
  3.15829,
  3.19888
)
num_other_scores = length(scores)-1

data = data.table(
  Team=factor(teams, levels=teams),
  Error=scores,
  Fill="grey"
)
data$ScoreText <- sprintf("%.2f", data$Error)

p <- ggplot(data, aes(x=Team, y=Error, fill=as.factor(Team),
                      label=ScoreText)) +
  geom_bar(stat="identity") +
  geom_text(vjust=-0.25) +
  ylim(0, max(scores)) +
  scale_fill_manual(values = c(highlight_col, rep("grey", num_other_scores))) +
  theme(legend.position="none", axis.text.x = element_text(
    angle = 45, vjust = 0.5)) +
  coord_cartesian(ylim = c(min_y, max(scores)+0.1))
print(p)
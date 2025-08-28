library(ggplot2)
library(RColorBrewer)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)


data1 <- read.csv(file.path(folder, '..', 'scripts', 'runs', 'detect', 'train', 
                           'results.csv'))
data1$season <- "may"
data1 <- subset(data1, epoch <= 50)

data2 <- read.csv(file.path(folder, '..', 'results', 'runs', 'detect', 'train', 
                            'results.csv'))
data2$season <- "jan"

data <- rbind(data1, data2)
data$season <- factor(data$season,
   levels = c('may', 'jan'),
   labels = c('Spring', 'Winter'))

data$discrete <- cut(data$epoch, seq(0,100,2))
data$continuous = ""
data$continuous[data$discrete == '(0,2]'] <- 1
data$continuous[data$discrete == '(2,4]'] <- 3
data$continuous[data$discrete == '(4,6]'] <- 5
data$continuous[data$discrete == '(6,8]'] <- 7
data$continuous[data$discrete == '(8,10]'] <- 9
data$continuous[data$discrete == '(10,12]'] <- 10
data$continuous[data$discrete == '(12,14]'] <- 13
data$continuous[data$discrete == '(14,16]'] <- 15
data$continuous[data$discrete == '(16,18]'] <- 17
data$continuous[data$discrete == '(18,20]'] <- 19
data$continuous[data$discrete == '(20,22]'] <- 21
data$continuous[data$discrete == '(22,24]'] <- 23
data$continuous[data$discrete == '(24,26]'] <- 25
data$continuous[data$discrete == '(26,28]'] <- 27
data$continuous[data$discrete == '(28,30]'] <- 29
data$continuous[data$discrete == '(30,32]'] <- 31
data$continuous[data$discrete == '(32,34]'] <- 33
data$continuous[data$discrete == '(34,36]'] <- 35
data$continuous[data$discrete == '(36,38]'] <- 37
data$continuous[data$discrete == '(38,40]'] <- 39
data$continuous[data$discrete == '(40,42]'] <- 41
data$continuous[data$discrete == '(42,44]'] <- 43
data$continuous[data$discrete == '(44,46]'] <- 45
data$continuous[data$discrete == '(46,48]'] <- 47
data$continuous[data$discrete == '(48,50]'] <- 49
data$continuous[data$discrete == '(50,52]'] <- 50
data$continuous[data$discrete == '(52,54]'] <- 53
data$continuous[data$discrete == '(54,56]'] <- 55
data$continuous[data$discrete == '(56,58]'] <- 57
data$continuous[data$discrete == '(58,60]'] <- 59
data$continuous[data$discrete == '(60,62]'] <- 61
data$continuous[data$discrete == '(62,64]'] <- 63
data$continuous[data$discrete == '(64,66]'] <- 65
data$continuous[data$discrete == '(66,68]'] <- 67
data$continuous[data$discrete == '(68,70]'] <- 69
data$continuous[data$discrete == '(70,72]'] <- 71
data$continuous[data$discrete == '(72,74]'] <- 73
data$continuous[data$discrete == '(74,76]'] <- 75
data$continuous[data$discrete == '(76,78]'] <- 77
data$continuous[data$discrete == '(78,80]'] <- 79
data$continuous[data$discrete == '(80,82]'] <- 81
data$continuous[data$discrete == '(82,84]'] <- 83
data$continuous[data$discrete == '(84,86]'] <- 85
data$continuous[data$discrete == '(86,88]'] <- 87
data$continuous[data$discrete == '(88,90]'] <- 89
data$continuous[data$discrete == '(90,92]'] <- 91
data$continuous[data$discrete == '(92,94]'] <- 93
data$continuous[data$discrete == '(94,96]'] <- 95
data$continuous[data$discrete == '(96,98]'] <- 97
data$continuous[data$discrete == '(98,100]'] <-99

#####################
## TRAINING LOSSES ##
#####################
df2 <- data[, c("continuous", "season", "train.cls_loss", "val.cls_loss", 
                "metrics.precision.B.", "metrics.recall.B.", "metrics.mAP50.B.")]

df2 <- df2 %>%
  pivot_longer(cols = c(train.cls_loss, val.cls_loss,  metrics.precision.B., 
                        metrics.recall.B., metrics.mAP50.B.), names_to = "metrics", 
               values_to = "values")

df3 <- df2 %>%
  filter(metrics %in% c("train.cls_loss", "val.cls_loss")) %>%
  mutate(metrics = recode(metrics,
                          "train.cls_loss" = "Training loss",
                          "val.cls_loss" = "Validation loss"))

df3$continuous = as.numeric(df3$continuous)
df3 = df3 %>%
  group_by(metrics, season, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

yolo_losses <- ggplot(df3, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.5) +
  labs( colour = NULL, title = "A",x = "Epochs", y = "Losses",
        fill = "Learning Curves") + 
  scale_color_manual(values = c("red", "blue")) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(fill = guide_legend(nrow = 2)) +
  facet_wrap( ~ season, ncol = 2)

#############
## METRICS ##
#############

df4 <- df2 %>%
  filter(metrics %in% c("metrics.precision.B.", "metrics.recall.B.")) %>%
  mutate(metrics = recode(metrics,
                          "metrics.precision.B." = "Precision",
                          "metrics.recall.B." = "Recall"))

df4$continuous = as.numeric(df4$continuous)
df4 = df4 %>%
  group_by(metrics, season, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

cls_metrics <- ggplot(df4, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.5) +
  labs( colour = NULL, title = "B",x = "Epochs", y = "Scores",
        fill = "Classifier Metrics") +
  scale_color_manual(values = c("red", "blue")) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(fill = guide_legend(nrow = 2)) + 
  facet_wrap( ~ season, ncol = 2)

##############
## F1-SCORE ##
##############
df5 <- data[, c("continuous", "season", "metrics.precision.B.", "metrics.recall.B.")]

df5 <- df5 %>%
  mutate(f_score = 2 * (metrics.precision.B. * metrics.recall.B.) /
           (metrics.precision.B. + metrics.recall.B.))

df5 = select(df5, f_score, continuous, season)
df5$continuous = as.numeric(df5$continuous)
df5 = df5 %>%
  group_by(continuous, season) %>%
  summarise(
    mean = mean(f_score),
    sd = sd(f_score))

f1_score <- ggplot(df5, aes(continuous, mean, color = season)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.5) +
  labs( colour = NULL, title = "B",x = "Epochs", y = "Scores",
        fill = "Classifier Metrics") +
  scale_color_manual(values = c("red", "blue")) +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(fill = guide_legend(nrow = 2)) 

#################
## Panel Plots ##
#################
metric_plots <- ggarrange(yolo_losses, cls_metrics, f1_score, ncol = 2, 
                          nrow = 2, common.legend = FALSE) 

path = file.path(folder, 'figures', 'comparative_metrics.png')
png(path, units = "in", width = 8, height = 6, res = 300)
print(metric_plots)
dev.off()


####################
## ACCURACY PLOTS ##
####################












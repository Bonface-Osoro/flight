library(ggpubr)
library(ggplot2)
library(tidyverse)
library(ggtext)
library(readr)
library(RColorBrewer)
library(dplyr)
library("cowplot")

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

###################
## TRAINING LOSS ##
###################

data <- read.csv(file.path(folder, '..', 'scripts', 'runs', 'detect', 'train', 
                           'results.csv'))

data$discrete <- cut(data$epoch, seq(0,50,2))
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

df = select(data, train.cls_loss,continuous)
df$continuous = as.numeric(df$continuous)
df = df %>%
  group_by(continuous) %>%
  summarise(
    mean = mean(train.cls_loss),
    sd = sd(train.cls_loss))

ggplot(df, aes(continuous, mean)) + 
  geom_line(aes(linetype = "Results"), position = position_dodge(width = 0.5), 
            size = 0.5, color = 'red') +
  geom_smooth(aes(linetype = "Smooth"), 
              method = "loess", 
              se = FALSE, 
              color = "blue", 
              size = 0.7) +
  labs(linetype = 'Lines', title = "A", x = "Epochs", y = "Loss") +
  scale_linetype_manual(values = c("Results" = "solid", "Smooth" = "dotted")) +
  scale_color_brewer(palette = "Dark2") +
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(color = guide_legend(ncol = 6))

############
## LOSSES ##
############
df2 <- data[, c("continuous", "train.cls_loss", "val.cls_loss", 
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
  group_by(metrics, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

yolo_losses <- ggplot(df3, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.5) +
  labs( colour = NULL, title = "A",x = "Epochs", y = "Losses",
        fill = "Learning Curves") + 
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(color = guide_legend(ncol = 6))

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
  group_by(metrics, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

cls_metrics <- ggplot(df4, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.5) +
  labs( colour = NULL, title = "B",x = "Epochs", y = "Losses",
        fill = "Classifier Metrics") +
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(color = guide_legend(ncol = 6))

##############
## F1-SCORE ##
##############
df5 <- data[, c("continuous", "metrics.precision.B.", "metrics.recall.B.")]

df5 <- df5 %>%
  mutate(f_score = 2 * (metrics.precision.B. * metrics.recall.B.) /
           (metrics.precision.B. + metrics.recall.B.))

df5 = select(df5, f_score,continuous)
df5$continuous = as.numeric(df5$continuous)
df5 = df5 %>%
  group_by(continuous) %>%
  summarise(
    mean = mean(f_score),
    sd = sd(f_score))

yolo_f1_score <- ggplot(df5, aes(continuous, mean)) + 
  geom_line(aes(linetype = "Results"), position = position_dodge(width = 0.5), 
            size = 0.5, color = 'red') +
  geom_smooth(aes(linetype = "Smooth"), method = "loess", se = FALSE, 
              color = "blue", size = 0.7) + 
  labs(linetype = 'Lines', title = "C", x = "Epochs", y = "F1") +
  scale_linetype_manual(values = c("Results" = "solid", "Smooth" = "dotted")) +
  scale_color_brewer(palette = "Dark2") +
  theme(
    legend.position = 'bottom',
    axis.text.x = element_text(size = 10),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 15, face = "bold"),
    plot.subtitle = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    axis.title.y = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 10)) +
  guides(color = guide_legend(ncol = 6))








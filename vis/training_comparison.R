library(ggplot2)
library(ggpubr)
library(dplyr)
library(RColorBrewer)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

##############################
## TEMPORAL MODEL TRAINING ###
##############################
all_train <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                           'all_results.csv'))
all_train$season <- "Spring and Winter"
all_train$comparison <- "ALL"

january <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                                'jan_results.csv'))
january$season <- "Winter"
january$comparison <- "Season"

may <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                              'may_results.csv'))
may$season <- "Spring"
may$comparison <- "Season"

data <- rbind(all_train, january, may)

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


########################
## 1. Training losses ##
########################
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

df3$season <- factor(df3$season,
    levels = c('Spring', 'Winter', 'Spring and Winter'),
    labels = c('Spring', 'Winter', 'Spring and Winter'))

season_losses <- ggplot(df3, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.1) +
  labs( colour = NULL, title = "(A) Training Losses.", 
        subtitle = "Model performance based on temporal training data (landmine drone images).", 
        x = "Training Epochs", y = "Losses", fill = "Learning Curves") + 
  scale_color_manual(values = c("red", "blue")) +
  theme(
    legend.position = "right",
    legend.key.size = unit(0.3, "cm"),
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 4, face = "bold"),
    plot.subtitle = element_text(size = 3.5),
    axis.text.y = element_text(size = 2.5),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 2.8),
    legend.text = element_text(size = 2.5),
    strip.text = element_text(size = 2.8),
    axis.ticks.y = element_line(linewidth = 0.08),
    axis.ticks.x = element_line(linewidth = 0.08),
    axis.title.x = element_text(size = 3)) +
  guides(color = guide_legend(ncol = 1, title = 'Model \nPerformance')) + 
  facet_wrap( ~ season, ncol = 3)

#########################
## 2. Seasonal metrics ##
#########################

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

df4$season <- factor(df4$season,
   levels = c('Spring', 'Winter', 'Spring and Winter'),
   labels = c('Spring', 'Winter', 'Spring and Winter'))

cls_metrics <- ggplot(df4, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.1) +
  labs(colour = NULL, title = "(B) Model Precision and Recall.",x = "Epochs", y = "Scores",
      subtitle = "Model performance based on temporal training data (landmine drone images).",  
        fill = "Classifier Metrics") +
  scale_color_manual(values = c("red", "blue")) +
  scale_y_continuous(limits = c(0, 1)) +
  theme(
    legend.position = "right",
    legend.key.size = unit(0.3, "cm"),
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 4., face = "bold"),
    plot.subtitle = element_text(size = 3.5),
    axis.text.y = element_text(size = 2.5),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 2.8),
    legend.text = element_text(size = 2.5),
    strip.text = element_text(size = 2.8),
    axis.ticks.y = element_line(linewidth = 0.08),
    axis.ticks.x = element_line(linewidth = 0.08),
    axis.title.x = element_text(size = 3)) +
  guides(color = guide_legend(ncol = 1, title = 'Evaluation \nMetric')) + 
  facet_wrap( ~ season, ncol = 3)


##########################
## Temporal Panel Plots ##
##########################
metric_plots <- ggarrange(season_losses, cls_metrics, 
     nrow = 2, common.legend = FALSE) 

path = file.path(folder, 'figures', 'seasonal_comparative_metrics.png')
png(path, units = "in", width = 2.8, height = 2.2, res = 720)
print(metric_plots)
dev.off()


################################
## RADIOMETRIC MODEL TRAINING ##
################################
all_train <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                                'all_results.csv'))

all_train$band <- "RGB and LWIR"

rgb_train <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                              'rgb_results.csv'))
rgb_train$band <- "RGB"

lwir_train <- read.csv(file.path(folder, '..', 'results', 'model_training', 
                          'lwir_results.csv'))
lwir_train$band <- "LWIR"

data1 <- rbind(all_train, rgb_train, lwir_train)

data1$discrete <- cut(data1$epoch, seq(0,50,2))
data1$continuous = ""
data1$continuous[data1$discrete == '(0,2]'] <- 1
data1$continuous[data1$discrete == '(2,4]'] <- 3
data1$continuous[data1$discrete == '(4,6]'] <- 5
data1$continuous[data1$discrete == '(6,8]'] <- 7
data1$continuous[data1$discrete == '(8,10]'] <- 9
data1$continuous[data1$discrete == '(10,12]'] <- 10
data1$continuous[data1$discrete == '(12,14]'] <- 13
data1$continuous[data1$discrete == '(14,16]'] <- 15
data1$continuous[data1$discrete == '(16,18]'] <- 17
data1$continuous[data1$discrete == '(18,20]'] <- 19
data1$continuous[data1$discrete == '(20,22]'] <- 21
data1$continuous[data1$discrete == '(22,24]'] <- 23
data1$continuous[data1$discrete == '(24,26]'] <- 25
data1$continuous[data1$discrete == '(26,28]'] <- 27
data1$continuous[data1$discrete == '(28,30]'] <- 29
data1$continuous[data1$discrete == '(30,32]'] <- 31
data1$continuous[data1$discrete == '(32,34]'] <- 33
data1$continuous[data1$discrete == '(34,36]'] <- 35
data1$continuous[data1$discrete == '(36,38]'] <- 37
data1$continuous[data1$discrete == '(38,40]'] <- 39
data1$continuous[data1$discrete == '(40,42]'] <- 41
data1$continuous[data1$discrete == '(42,44]'] <- 43
data1$continuous[data1$discrete == '(44,46]'] <- 45
data1$continuous[data1$discrete == '(46,48]'] <- 47
data1$continuous[data1$discrete == '(48,50]'] <- 49

#############################
## 4. Band Training losses ##
#############################
df6_band <- data1[, c("continuous", "band", "train.cls_loss", "val.cls_loss", 
                "metrics.precision.B.", "metrics.recall.B.", "metrics.mAP50.B.")]

df6_band <- df6_band %>%
  pivot_longer(cols = c(train.cls_loss, val.cls_loss,  metrics.precision.B., 
                        metrics.recall.B., metrics.mAP50.B.), names_to = "metrics", 
               values_to = "values")

df6 <- df6_band %>%
  filter(metrics %in% c("train.cls_loss", "val.cls_loss")) %>%
  mutate(metrics = recode(metrics,
                          "train.cls_loss" = "Training loss",
                          "val.cls_loss" = "Validation loss"))

df6$continuous = as.numeric(df6$continuous)
df6 = df6 %>%
  group_by(metrics, band, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

df6$band <- factor(df6$band,
   levels = c('RGB', 'LWIR', 'RGB and LWIR'),
   labels = c('RGB', 'LWIR', 'RGB and LWIR'))

band_losses <- ggplot(df6, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.1) +
  labs( colour = NULL, title = "(A) Training Losses.", 
        subtitle = "Model performance based on training drone images taken at different radio bands (multiwavelength).", 
        x = "Training Epochs", y = "Losses", fill = "Learning Curves") + 
  scale_color_manual(values = c("red", "blue")) +
  theme(
    legend.position = "right",
    legend.key.size = unit(0.3, "cm"),
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 4., face = "bold"),
    plot.subtitle = element_text(size = 3.5),
    axis.text.y = element_text(size = 2.5),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 2.8),
    legend.text = element_text(size = 2.5),
    strip.text = element_text(size = 2.8),
    axis.ticks.y = element_line(linewidth = 0.08),
    axis.ticks.x = element_line(linewidth = 0.08),
    axis.title.x = element_text(size = 3)) +
  guides(color = guide_legend(ncol = 1, title = 'Model \nPerformance')) + 
  facet_wrap( ~ band, ncol = 3)

###########################
## 5. Radio band metrics ##
###########################

df7 <- df6_band %>%
  filter(metrics %in% c("metrics.precision.B.", "metrics.recall.B.")) %>%
  mutate(metrics = recode(metrics,
                          "metrics.precision.B." = "Precision",
                          "metrics.recall.B." = "Recall"))

df7$continuous = as.numeric(df7$continuous)
df7 = df7 %>%
  group_by(metrics, band, continuous) %>%
  summarise(
    mean = mean(values),
    sd = sd(values))

df7$band <- factor(df7$band,
           levels = c('RGB', 'LWIR', 'RGB and LWIR'),
           labels = c('RGB', 'LWIR', 'RGB and LWIR'))

band_cls_metrics <- ggplot(df7, aes(continuous, mean, color = metrics)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.1) +
  labs(colour = NULL, title = "(B) Model Precision and Recall.",x = "Training Epochs", y = "Scores",
       subtitle = "Model performance based on training drone images taken at different radio bands (multiwavelength).",  
       fill = "Classifier Metrics") +
  scale_color_manual(values = c("red", "blue")) +
  scale_y_continuous(limits = c(0, 1)) +
  theme(
    legend.position = "right",
    legend.key.size = unit(0.3, "cm"),
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 4, face = "bold"),
    plot.subtitle = element_text(size = 3.5),
    axis.text.y = element_text(size = 2.5),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 2.8),
    legend.text = element_text(size = 2.5),
    strip.text = element_text(size = 2.8),
    axis.ticks.y = element_line(linewidth = 0.08),
    axis.ticks.x = element_line(linewidth = 0.08),
    axis.title.x = element_text(size = 3)) +
  guides(color = guide_legend(ncol = 1, title = 'Evaluation \nMetrics')) + 
  facet_wrap( ~ band, ncol = 3)

##########################
## Temporal Panel Plots ##
##########################
band_metric_plots <- ggarrange(band_losses, band_cls_metrics, 
                          nrow = 2, common.legend = FALSE) 

path = file.path(folder, 'figures', 'band_comparative_metrics.png')
png(path, units = "in", width = 2.8, height = 2.2, res = 1080)
print(band_metric_plots)
dev.off()

#################
## 6. F1-Score ##
#################
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

df5$season <- factor(df5$season,
                     levels = c('Spring', 'Winter', 'Spring and Winter'),
                     labels = c('Spring', 'Winter', 'Spring and Winter'))

df5$comparison <- "Temporal"
names(df5)[names(df5) == "season"] <- "image_data"

df8 <- data1[, c("continuous", "band", "metrics.precision.B.", "metrics.recall.B.")]

df8 <- df8 %>%
  mutate(f_score = 2 * (metrics.precision.B. * metrics.recall.B.) /
           (metrics.precision.B. + metrics.recall.B.))

df8 = select(df8, f_score, continuous, band)
df8$continuous = as.numeric(df8$continuous)

df8 = df8 %>%
  group_by(continuous, band) %>%
  summarise(
    mean = mean(f_score),
    sd = sd(f_score))

df8$band <- factor(df8$band,
   levels = c('RGB', 'LWIR', 'RGB and LWIR'),
   labels = c('RGB', 'LWIR', 'RGB \nand LWIR'))

df8$comparison <- "Multiwavelength"
names(df8)[names(df8) == "band"] <- "image_data"

dff <- rbind(df5, df8)


dff$image_data <- factor(dff$image_data,
   levels = c("RGB", "LWIR", "RGB \nand LWIR", "Spring", "Winter", "Spring and Winter"),
   labels = c("RGB", "LWIR", "RGB \nand LWIR", "Spring", "Winter", "Spring \nand Winter"))

comparative_f1_score <- ggplot(dff %>%filter(
    (comparison == "Multiwavelength" & image_data %in% c("RGB", "LWIR", "RGB \nand LWIR")) |
    (comparison == "Temporal" & image_data %in% c("Spring", "Winter", "Spring \nand Winter"))),
  aes(continuous, mean, color = image_data)) + 
  geom_line(position = position_dodge(width = 0.5), size = 0.25) +
  labs(colour = NULL,  title = "Calculated F1 Score.", x = "Training Epochs", y = "Scores",
    subtitle = "F1 score during training and validation on landmine images at different bands (Multiwavelength) and seasons (Temporal).",
    fill = "Classifier Metrics") + 
  scale_color_brewer(palette = "Dark2") +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 3),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 4.5, face = "bold"),
    plot.subtitle = element_text(size = 3.5),
    axis.text.y = element_text(size = 3),
    axis.title.y = element_text(size = 3.5),
    legend.title = element_text(size = 3.5),
    legend.text = element_text(size = 3),
    axis.ticks.y = element_line(linewidth = 0.1),
    axis.ticks.x = element_line(linewidth = 0.1),
    strip.text = element_text(size = 3.5),
    axis.title.x = element_text(size = 3.5)) +
  guides(color = guide_legend(ncol = 6, title = 'Source \nImage')) +
  facet_wrap(~ comparison, ncol = 2, scales = "free_x", drop = TRUE)

path = file.path(folder, 'figures', 'comparative_f1_score.png')
png(path, units = "in", width = 3, height = 2.5, res = 720)
print(comparative_f1_score)
dev.off()

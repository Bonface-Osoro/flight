library(ggplot2)
library(ggpubr)
library(dplyr)
library(RColorBrewer)

suppressMessages(library(tidyverse))
folder <- dirname(rstudioapi::getSourceEditorContext()$path)

####################
## MEAN MODEL mAP ##
####################
###################################
## 1. Performance on Self Images ##
###################################
all_map <- read.csv(file.path(folder, '..', 'results', 'all_images_metrics.csv'))
rgb_map <- read.csv(file.path(folder, '..', 'results', 'Multiwavelength', 
                              'rgb_images_metrics.csv'))
lwir_map <- read.csv(file.path(folder, '..', 'results', 'Multiwavelength', 
                              'lwir_images_metrics.csv'))

data <- rbind(all_map, rgb_map, lwir_map)
names(data)[names(data) == "band"] <- "image_data"
data$comparison <- "Multiwavelength"

all_map2 <- read.csv(file.path(folder, '..', 'results', 'all_images_metrics.csv')) %>%
  dplyr::select(-band)

all_map2$season <- "Spring and Winter"

jan_map <- read.csv(file.path(folder, '..', 'results', 'Temporal', 
                              'jan_images_metrics.csv'))

may_map <- read.csv(file.path(folder, '..', 'results', 'Temporal', 
                              'may_images_metrics.csv'))

data1 <- rbind(all_map2, jan_map, may_map)
names(data1)[names(data1) == "season"] <- "image_data"
data1$comparison <- "Temporal"

data3 <- rbind(data, data1)

df <- data3 %>%
  group_by(image_data, comparison) %>%
  summarize(mean = mean(ap50, na.rm = TRUE) * 100,
            sd   = sd(ap50, na.rm = TRUE) * 100,
            n    = n(),
            .groups = "drop")

df$image_data <- as.character(df$image_data)
label_map <- c("optical" = "RGB", "infrared" = "LWIR", "all" = "RGB and LWIR",
  "spring" = "Spring", "winter" = "Winter", "Spring and Winter" = "Spring and Winter")

df <- df %>%
  mutate(image_data = dplyr::recode(image_data, !!!label_map))

desired_order <- c("RGB", "LWIR", "RGB and LWIR", "Spring", "Winter", "Spring and Winter")
present_levels <- intersect(desired_order, unique(df$image_data))
df$image_data <- factor(df$image_data, levels = present_levels)

df <- droplevels(df)

individual_model <- ggplot(df, aes(x = image_data, y = mean, fill = image_data)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.98) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd),
                width = .2, position = position_dodge(.9),
                color = 'red', size = 0.2) +
  geom_text(aes(label = formatC(signif(mean, 4), digits = 3, format = "fg")),
            color = 'black', size = 1.2, position = position_dodge(0.9),
            vjust = -0.5, hjust = -0.2, angle = 0) +
  scale_fill_viridis_d(direction = -1, guide = "none") +
  labs(title = "(A) Model Performance on Self Images.",
       subtitle = "Individual model performance on the training images that correspond to the radio bands (multiwavelength) \nor period (temporal) the model was trained on.",
       x = NULL, y = "mAP (%)") +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 5, face = "bold"),
    plot.subtitle = element_text(size = 4),
    axis.text.y = element_text(size = 3),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 3),
    legend.text = element_text(size = 3),
    axis.ticks.y = element_line(linewidth = 0.1),
    axis.ticks.x = element_line(linewidth = 0.1),
    strip.text = element_text(size = 4),
    axis.title.x = element_text(size = 5)) +
  expand_limits(y = 0) + 
  scale_x_discrete(drop = TRUE) +
  scale_y_continuous(expand = c(0, 0),
  labels = function(y) format(y, scientific = FALSE), limits = c(0, 105)) +
  facet_wrap(~ comparison, ncol = 2, scales = "free_x", drop = TRUE)

##################################
## 2. Performance on New Images ##
##################################
d1 <- read.csv(file.path(folder, '..', 'validation', 
                         'jan_model_transferred_on_may_images_metrics.csv'))

d2 <- read.csv(file.path(folder, '..', 'validation', 
                         'lwir_model_transferred_on_rgb_images_metrics.csv'))

d3 <- read.csv(file.path(folder, '..', 'validation', 
                         'may_model_transferred_on_jan_images_metrics.csv'))

d4 <- read.csv(file.path(folder, '..', 'validation', 
                         'rgb_model_transferred_on_lwir_metrics.csv'))

data1 <- rbind(d1, d2, d3, d4)
data1$comparison <- "Individual Models"
data1renamed <- data1

data1renamed$model <- factor(data1renamed$model,
    levels = c('RGB on LWIR Images', 'LWIR on RGB Images', 'Jan on May Images', 'May on Jan Images'),
    labels = c('RGB Model \nTested on \nLWIR Images', 'LWIR Model \nTested on \nRGB Images', 
               'Winter Model \nTested on \nSpring Images', 'Spring Model \nTested on \nWinter Images'))

d5 <- read.csv(file.path(folder, '..', 'validation', 
                         'ALL_model_on_jan_images_metrics.csv'))

d6 <- read.csv(file.path(folder, '..', 'validation', 
                         'ALL_model_on_LWIR_images_metrics.csv'))

d7 <- read.csv(file.path(folder, '..', 'validation', 
                         'ALL_model_on_may_images_metrics.csv'))

d8 <- read.csv(file.path(folder, '..', 'validation', 
                         'ALL_model_on_RGB_images_metrics.csv'))

data2 <- rbind(d5, d6, d7, d8)
data2$comparison <- "General Model"

data2$model <- factor(data2$model,
    levels = c('ALL on RGB Images', 'ALL on LWIR Images', 'ALL on Jan Images', 'ALL on May Images'),
    labels = c('General \nModel Tested \non RGB Images', 'General \nModel Tested \non LWIR Images', 
               'General \nModel Tested \non Winter Images', 'General \nModel Tested \non Spring Images'))

data4 <- dplyr::bind_rows(
  data1renamed %>% mutate(comparison = "Individual Models",
                   model = as.character(model)),
  data2 %>% mutate(comparison = "General Model",
                   model = as.character(model)))

data4 <- data4 %>%
  mutate(model = trimws(as.character(model)),
         comparison = trimws(as.character(comparison)))

df4 <- data4 %>%
  group_by(comparison, model) %>%
  summarize(mean = mean(map50, na.rm = TRUE) * 100,
            sd   = sd(map50, na.rm = TRUE) * 100, n    = n(),.groups = "drop")

desired_order <- c('RGB Model \nTested on \nLWIR Images', 'LWIR Model \nTested on \nRGB Images',
   'Winter Model \nTested on \nSpring Images', 'Spring Model \nTested on \nWinter Images',
   'General \nModel Tested \non RGB Images', 'General \nModel Tested \non LWIR Images',
   'General \nModel Tested \non Winter Images', 'General \nModel Tested \non Spring Images')

present_levels <- desired_order[desired_order %in% unique(df4$model)]
df4 <- df4 %>% mutate(model = factor(model, levels = present_levels))

df4$comparison <- factor(df4$comparison, levels = unique(df4$comparison))

model_comparion_average <- ggplot(df4, aes(x = model, y = mean, fill = model)) +
  geom_col(width = 0.98, color = NA) +
  geom_errorbar(aes(ymin = pmax(0, mean - 0.6*sd), ymax = mean + 0.5*sd),
                width = 0.2, color = "red", size = 0.2) +
  geom_text(aes(label = sprintf("%.1f", mean)), size = 0.8, vjust = -0.3, 
            hjust = -0.2, angle = 0) +
  labs(title = "(B) Model Performance on New Images.",
       subtitle = "Overall mAP across all object classes (landmine types) on the training images taken at different radio \nbands (multiwavelength) and period of the year (temporal).",
       x = NULL, y = "mAP (%)") +
  scale_fill_viridis_d(direction = -1) + 
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 2.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 5, face = "bold"),
    plot.subtitle = element_text(size = 4),
    axis.text.y = element_text(size = 3),
    axis.title.y = element_text(size = 3),
    legend.title = element_text(size = 3),
    legend.text = element_text(size = 3),
    axis.ticks.y = element_line(linewidth = 0.1),
    axis.ticks.x = element_line(linewidth = 0.1),
    strip.text = element_text(size = 4),
    axis.title.x = element_text(size = 5)) +
  expand_limits(y = 0) +
  scale_x_discrete(drop = TRUE) +
  scale_y_continuous(expand = c(0, 0),
                     labels = function(y) format(y, scientific = FALSE), limits = c(0, 105)) +
  facet_wrap(~ comparison, ncol = 2, scales = "free_x", drop = TRUE)

##############################
## Combined mAP Panel Plots ##
##############################
comparative_map_plots <- ggarrange(individual_model, model_comparion_average, nrow = 2,
                                   common.legend = FALSE) 

path = file.path(folder, 'figures', 'comparative_map.png')
png(path, units = "in", width = 3, height = 3.5, res = 720)
print(comparative_map_plots)
dev.off()

###################
## MODEL TESTING ##
###################
df2 = data1 %>%
  group_by(model, class) %>%
  summarize(mean = mean(map50)*100,
            sd = sd(map50)*100)

df2$class <- factor(df2$class,
     levels = c('ap_metal', 'ap_plastic', 'at_metal', 'at_plastic'),
     labels = c('AP Metal', 'AP Plastic', 'AT Metal', 'AT Plastic'))

df2$model <- factor(df2$model,
  levels = c('RGB on LWIR Images', 'LWIR on RGB Images', 'Jan on May Images', 'May on Jan Images'),
  labels = c('RGB \nModel Tested \non LWIR Images', 'LWIR \nModel Tested \non RGB Images', 
             'Winter \nModel Tested \non Spring Images', 'Spring \nModel Tested \non Winter Images'))

df2$comparison <- "Individual Models"

df3 = data2 %>%
  group_by(model, class) %>%
  summarize(mean = mean(map50)*100,
            sd = sd(map50)*100)

df3$class <- factor(df3$class,
    levels = c('ap_metal', 'ap_plastic', 'at_metal', 'at_plastic'),
    labels = c('AP Metal', 'AP Plastic', 'AT Metal', 'AT Plastic'))

df3$comparison <- "General Model"

dff <- rbind(df2, df3)

model_class_test <- ggplot(dff, aes(x = model, y = mean, fill = class)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.9) +
  geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = .2,
                position = position_dodge(.9), color = 'red',size = 0.3) + 
  geom_text(aes(label = formatC(signif(after_stat(y), 4), 
     digits = 3, format = "fg", flag = "#")), color = 'black', size = 2, position = 
      position_dodge(0.9), vjust = 0.5, hjust = -0.2, angle = 90) +
  scale_fill_viridis_d(direction = -1) + 
  labs(colour = NULL, title = "Mean Average Precision (mAP).", 
       subtitle = "Recorded mAP for models trained on different images and tested on new sets of images by object classes (landmine types).",
       x = NULL, y = "mAP (%)") + 
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(size = 4.5),
    panel.spacing = unit(0.6, "lines"),
    plot.title = element_text(size = 9, face = "bold"),
    plot.subtitle = element_text(size = 6),
    axis.text.y = element_text(size = 4.5),
    axis.title.y = element_text(size = 5),
    legend.title = element_text(size = 6),
    legend.text = element_text(size = 5),
    axis.ticks.y = element_line(linewidth = 0.2),
    axis.ticks.x = element_line(linewidth = 0.2),
    strip.text = element_text(size = 6),
    axis.title.x = element_text(size = 5)) +
  guides(fill = guide_legend(ncol = 7, title = 'Landmine Type')) +
  scale_x_discrete(expand = c(0, 0.15)) +
  scale_y_continuous(expand = c(0, 0),
  labels = function(y) format(y, scientific = FALSE),limits = c(0, 110)) +
  facet_wrap(~ comparison, ncol = 2, scales = "free_x", drop = TRUE)

path = file.path(folder, 'figures', 'object_class_map.png')
png(path, units = "in", width = 5, height = 3.5, res = 720)
print(model_class_test)
dev.off()

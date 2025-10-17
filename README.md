# Fusion-based Learning and Inference for Ground and High-altitude Targets (Flight)
With the growing availability of affordable drones and sensor technologies, high-resolution RGB and thermal (LWIR) imagery can now be collected safely and efficiently over contaminated areas. Converting this data into actionable insights requires automated AI-based methods capable of detecting and classifying objects under varying illumination, seasons, and spectral conditions. This project leverages Convolutional Neural Networks (CNNs), specifically YOLOv8 to investigate how models trained on one wavelength or season perform when transferred to another, aiming to improve landmine detection accuracy while reducing risk and manual intervention.

The workflow implements a transfer learning approach where pretrained YOLO models are fine-tuned on drone images captured in both the visible (RGB) and long-wave infrared (LWIR) bands. Experiments assess (1) cross-band transferability, (2) seasonal generalization, and (3) model convergence across image types. By comparing model performance on unseen spectral and temporal datasets, this project provides empirical evidence on the generalizability of pretrained object detection models in real-world drone applications. The repository includes data preprocessing scripts, training configurations, evaluation metrics (mAP, precision, recall), and visualization notebooks for reproducibility and adaptation to related geospatial detection tasks.



Methodology
==============
Here the transfer learning approach implemented in the study (see `Figure 1`). 

#### Figure 1 Proposed Method.
<p align="center">
  <img src="/docs/Method_Box.png" />
</p>

Results
=======
`Figure 2` shows the object detection results of the CNN models trained to identify different landmine types on images.
#### Figure 2 model performance on detection of various landmine types.
<p align="center">
  <img src="/docs/object_class_map.png" />
</p>
A side-to-side comparison of transfer pre-trained models on new sets of different images per landmine type (AP, anti-personnel and AT, anti-tank). 

## Background and funding

**flight** has been developed by researchers at the Digital Epidemiology (**De**)
laboratory of the **Digital futures** at University of Cincinnati and Geography and Geoinformation Science Department of George Mason University. 

The authors would like to thank the Department of Geography and Geoinformation Science at George Mason University for funding this work. We also thank the U.S. Army’s Counter Explosive Hazards Center (CEHC) for providing the real landmines used in this study. 

## Required Data
[1]	“Training images,” zenodo repository. https://www.worldpop.org/ (accessed October 17, 2025).
# Multimodal Machine Learning Models for Predicting Mode of Delivery After Labor Induction

 This project explores multimodal machine learning models for predicting the mode of delivery — vaginal delivery (VD) or cesarean section (CS) — following induction of labor (IOL).
 The models are trained on two data modalities: tabular maternal-fetal clinical data and third-trimester ultrasound images from three anatomical views (abdomen, head, and femur).
 Each model architecture (MM1, MM2, and MM3) results from Hyperparameter Optimization using AutoGluon-Multimodal Predictor (autoMM), applied per fold of a three-fold cross-validation split.
   
## Data
 This code expects CSV files as input, where:
  - Each row corresponds to a single patient
  - Tabular features and paths to ultrasound images in separate columns

 It follows the data input structure expected by the [AutoGluon Multimodal Pipeline](https://auto.gluon.ai/stable/tutorials/multimodal/multimodal-faq.html#:~:text=How%20does%20AutoGluon%20MultiModal%20handle%20multiple%20images%20per%20sample?).

 All scripts depend on a shared file named `columns_config.json`, which maps your dataset variables: **num_cols** (numerical features), **cat_cols** (categorical features) and **image_cols** (image paths columns). Update this file with your feature names before running the scripts.

 ### Data Availability
 The data are not made publicly available for reasons of privacy and ethical restrictions.

## Corresponding email:
  - baba.mrodriguess@gmail.com (for code related questions)

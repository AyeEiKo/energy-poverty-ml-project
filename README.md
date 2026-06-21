# Energy Poverty Prediction using Machine Learning and Deep Learning

## Project Overview
This project applies machine learning (ML) and deep learning (DL) techniques to predict multidimensional household energy poverty using Demographic and Health Survey (DHS) data. The study focuses on Myanmar, India, and Pakistan, as well as a combined cross-country dataset, to evaluate model generalisation and robustness across different national contexts. In addition, the project investigates the impact of wealth index inclusion on predictive performance and provides a comparative analysis of socioeconomic and demographic determinants of energy poverty.

## Research Objectives
- To compare the predictive performance of traditional machine learning and deep learning models for household  energy poverty classification across Myanmar, Pakistan, and India.
- To examine the impact of wealth index inclusion on the predictive performance of machine learning models for multidimensional energy poverty.
- To evaluate the stability and robustness of model performance using 10 repeated stratified train-test splits.
- To assess the statistical significance of performance differences using paired t-tests across experimental settings.

## Methodology
- Stratified 80/20 train-test splits (10 repeated runs)
- Models: LightGBM, CatBoost, Random Forest, Extra Trees, NN_TORCH, FastAI, FT-Transformer
- Evaluation metrics: Accuracy, MCC, Balanced Accuracy, F1-score
- Best-performing ML and DL models selected per dataset
- Statistical comparison using paired t-tests

## Datasets
- Myanmar DHS dataset
- India DHS dataset
- Pakistan DHS dataset
- Combined multi-country dataset

## Key Features
- Feature engineering for energy poverty indicators (MEPI-based)
- Wealth-inclusive and wealth-exclusive experimental settings
- Cross-country comparison of predictive performance
- Model evaluation on unseen test splits

## Results Summary
- Consistent performance across ML and DL models
- Wealth index significantly impacts predictive accuracy
- Different models (CatBoost, LightGBM, NN_TORCH, FastAI, and FT-Transformer) emerge as best-performing depending on dataset and feature setting

## Repository Structure
- `MYANMAR/` - Myanmar analysis and models
- `INDIA/` - India analysis and models
- `PAKISTAN/` - Pakistan analysis and models
- `COMBINED/` - Cross-country analysis
- `requirement_details.txt` - Dependencies

## Installation
```bash
pip install -r requirement_details.txt

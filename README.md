# Large Language Models for forecasting market behavior

## Overview

This repository contains the code and resources related to our Bachelor's thesis in Computer Science on market price forecasting using large language models (LLMs). The repository is structured into three main folders:

- **thesis**: LaTeX files for the thesis document.
- **simpler_models**:
  - **data_loader**: Data loading utilities.
  - **experiments** : files running the models.
  - **scripts**: Scripts for running the models.
  - **results**: Arguments and results of each model training.
  - **utils**: Utility functions.
  - **models**: Model implementations.
- **final_project**: Submodule with the final project repo.

## Simpler Models

The `simpler_models` folder includes implementations of various machine learning models aimed at market price forecasting:

- **MLP** (Multi-Layer Perceptron)
- **CNN** (Convolutional Neural Network)
- **ResNet** (Residual Network)
- **Linear Regression**

### Usage

models can be trained using the scripts in the `scripts` folder. For example, to train an MLP model, run the following command:

```bash
bash ./scripts/MLP_US500USD.sh
```

## Final Project: Trading-LLM

The `final_project` focuses on reprogramming a Large Language Model (LLM) for time series forecasting. More information can be found in the submodule's README.

## Authors

This project was made possible thanks to the hard work and dedication of the following team members:

- [Damian Dąbrowski](https://github.com/damiad)
- [Ivan Gechu](https://github.com/ivgechu)
- [Heorhii Lopatin](mailto:teammate2.email@example.com)
- [Krzysztof Szostek](https://github.com/kamis12-bit)

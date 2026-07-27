# Master Thesis Code

## Title

**Quantifying the Value of Learning about Ocean Heat Uptake: A Cost-Risk Perspective on Ocean Parameter Uncertainty**

## Overview

This repository contains the computational code developed for my Master's thesis in Ocean and Climate Physics at the University of Hamburg.

The thesis investigates uncertainty in ocean heat uptake within a climate-economic modelling framework, with a focus on Bayesian learning, uncertainty quantification, and the value of information.

The optimization model is implemented in GAMS, while Python is used for parameter-grid generation, preprocessing, analysis of model outputs, and visualization.

## Repository Structure

- `gams/` – final GAMS model implementations used for the thesis analysis
- `python/` – Python scripts for parameter-grid generation, preprocessing, result analysis, and visualization

## Methods

The computational work includes:

- Bayesian learning under uncertain climate parameters
- transformation and reparameterization of prior uncertainty distributions
- Jacobian-based change-of-variables methods
- multidimensional integration
- treatment of Dirac delta distributions in the context of Bayesian learning
- climate-economic optimization in GAMS
- parameter-grid generation and numerical analysis in Python

## GAMS Models

The repository includes model implementations for:

- full learning
- partial learning

These models are used to evaluate climate-economic outcomes under uncertainty in ocean-related parameters.

## Python

Python scripts are used to:

- generate parameter grids for the GAMS models
- prepare model inputs
- process and analyse model outputs
- produce figures and visualizations used in the thesis analysis

## Results and Figures

Selected thesis results and figures are not included in this public repository due to planned journal publication.

The corresponding analysis and visualization scripts are included where appropriate.

## Software

- GAMS
- Python

## Status

The repository contains code associated with the completed Master's thesis and may be updated as the work is prepared for scientific publication.

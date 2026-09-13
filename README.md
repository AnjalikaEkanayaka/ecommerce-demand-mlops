# E-Commerce Demand Forecasting MLOps Platform

[![CI](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

A portfolio project for aggregate daily e-commerce demand forecasting using the
**Olist Brazilian E-Commerce Dataset**, XGBoost, FastAPI, MLflow and Evidently.
GitHub Actions runs tests and builds an image; it does not deploy the application.

The project is being corrected in verified phases. See [cleanup checkpoints and
current setup instructions](docs/progress.md). Drift monitoring currently uses
simulated data, and safe candidate promotion is not implemented yet. The diagram
below represents the intended lifecycle, not a deployed production system.

---

## Architecture & System Overview

```text
┌─────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ Raw E-Commerce  │ ───► │ Pydantic Data      │ ───► │ Time-Series        │
│ Kaggle Datasets │      │ Validation Engine  │      │ Feature Generator  │
└─────────────────┘      └────────────────────┘      └─────────┬──────────┘
                                                               │
                                                               ▼
┌─────────────────┐      ┌────────────────────┐      ┌────────────────────┐
│ Evidently AI    │ ◄─── │ FastAPI REST API   │ ◄─── │ XGBoost Time-Series│
│ Drift Monitor   │      │ Microservice       │      │ Forecast Model     │
└────────┬────────┘      └────────────────────┘      └─────────┬──────────┘
         │                                                     │
         ▼                                                     ▼
┌─────────────────┐                                  ┌────────────────────┐
│ Automated       │                                  │ MLflow Experiment  │
│ Retraining Loop │                                  │ Tracking Server    │
└─────────────────┘                                  └────────────────────┘

# E-Commerce Demand Forecasting MLOps Platform

[![CI/CD Pipeline](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/AnjalikaEkanayaka/ecommerce-demand-mlops/actions)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)](https://www.docker.com/)

An end-to-end, production-grade MLOps platform for real-time e-commerce demand forecasting. Built on the **Olist Brazilian E-Commerce Dataset**, this repository demonstrates automated data validation, time-series feature engineering, XGBoost forecasting, FastAPI serving, MLflow experiment tracking, Evidently AI drift monitoring, and GitHub Actions CI/CD automation.

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

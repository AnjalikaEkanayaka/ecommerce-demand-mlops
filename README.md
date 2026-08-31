# E-Commerce Dynamic Demand Forecasting & MLOps Pipeline

Production-ready MLOps framework for dynamic demand forecasting using real-world e-commerce transactional data.

## System Capabilities
- **ML Engine**: XGBoost multi-step demand forecaster.
- **Serving**: Asynchronous FastAPI service with Pydantic schema validation.
- **MLOps Infrastructure**: Experiment tracking & registry via MLflow.
- **Quality & Monitoring**: Automated testing (`pytest`) and data drift monitoring (`Evidently AI`).
- **Containerization & CI/CD**: Dockerized microservice orchestrated via GitHub Actions.
# MLOps Project: Complete ML Pipeline with AWS Infrastructure

This repository contains a comprehensive MLOps project that demonstrates how to build, deploy, and manage a machine learning pipeline using AWS services, Terraform for Infrastructure as Code, and modern DevOps practices.

## 🎯 Project Overview

This project showcases a complete MLOps workflow for an insurance prediction model, progressing through different maturity levels:

- **MLOps_2**: Basic AWS infrastructure setup with Terraform
- **MLOps_3**: Enhanced infrastructure with modular Terraform and CI/CD
- **MLOps_4**: Complete ML pipeline with model training, deployment, and monitoring

## 📁 Project Structure

```
MLOps/
├── MLOps_2/          # Basic Terraform infrastructure setup
├── MLOps_3/          # Enhanced infrastructure with CI/CD
├── MLOps_4/          # Complete ML pipeline with model deployment
├── docs/             # Project documentation and diagrams
└── .github/          # GitHub Actions workflows
```

## 🚀 What You'll Learn

### Infrastructure as Code (IaC)
- **Terraform Configuration**: Modular, reusable infrastructure components
- **Multi-Environment Support**: Dev, test, and production environments
- **Remote State Management**: S3 backend with state locking and versioning
- **CI/CD Automation**: GitHub Actions for infrastructure deployment

### Machine Learning Pipeline
- **Data Ingestion**: Automated data loading and preprocessing
- **Model Training**: MLflow-based experiment tracking and model versioning
- **Model Deployment**: FastAPI application with containerization
- **Model Monitoring**: Performance tracking and model comparison

### DevOps Practices
- **Version Control**: Git-based workflow with branching strategies
- **Containerization**: Docker for consistent deployment environments
- **Testing**: Unit tests for ML pipeline components
- **Monitoring**: AWS CloudWatch integration

## 🛠️ Technology Stack

- **Infrastructure**: AWS (S3, ECR, App Runner), Terraform
- **ML Framework**: Scikit-learn, MLflow
- **API Framework**: FastAPI
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Data Versioning**: DVC
- **Testing**: pytest

## 📋 Prerequisites

- [Python 3.8+](https://www.python.org/downloads/)
- [Terraform](https://developer.hashicorp.com/terraform/downloads)
- [AWS CLI](https://aws.amazon.com/cli/)
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/)

## 🏗️ Getting Started

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MLOps
```

### 2. Set Up AWS Credentials
```bash
aws configure
```

### 3. Choose Your Starting Point

#### For Infrastructure Setup (MLOps_2/MLOps_3)
```bash
cd MLOps_3/terraform
terraform init -backend-config=backends/dev.conf
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

#### For Complete ML Pipeline (MLOps_4)
```bash
cd MLOps_4/src
pip install -r requirements.txt
python main.py  # Train the model
python app.py   # Start the API server
```

## 🔄 Maturity Levels

### Level 2: Basic Infrastructure
- S3 buckets for data storage
- Basic Terraform configuration
- Local state management

### Level 3: Enhanced Infrastructure
- Modular Terraform architecture
- Remote state backend (S3)
- Multi-environment support
- GitHub Actions CI/CD

### Level 4: Complete ML Pipeline
- MLflow experiment tracking
- Automated model training pipeline
- FastAPI model serving
- Containerized deployment
- Comprehensive testing

## 📊 ML Pipeline Components

### Data Pipeline
- **Ingestion**: Load data from various sources
- **Cleaning**: Preprocess and validate data
- **Feature Engineering**: Transform raw data into features

### Model Pipeline
- **Training**: Train models with hyperparameter tuning
- **Evaluation**: Assess model performance
- **Versioning**: Track model versions with MLflow
- **Deployment**: Serve models via REST API

### Monitoring
- **Performance Tracking**: Monitor model accuracy and drift
- **Logging**: Centralized logging with AWS CloudWatch
- **Alerts**: Automated notifications for issues

## 🔧 Configuration

Each environment has its own configuration:
- `environments/dev.tfvars` - Development environment
- `environments/tst.tfvars` - Testing environment  
- `environments/prd.tfvars` - Production environment

## 🧪 Testing

Run the test suite:
```bash
cd MLOps_4/src
pytest tests/
```

## 📈 Monitoring and Observability

- **MLflow**: Experiment tracking and model registry
- **AWS CloudWatch**: Infrastructure and application monitoring
- **GitHub Actions**: CI/CD pipeline monitoring
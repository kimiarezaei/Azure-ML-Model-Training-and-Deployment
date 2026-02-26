# End-to-End Machine Learning with Azure ML

This repository demonstrates a complete, end-to-end machine learning workflow using Azure Machine Learning (Azure ML), from data ingestion and model training to deployment and real-time inference using managed online endpoints.

It showcases modern ML practices with CatBoost and ONNX for optimized, framework-independent model deployment.

---

## What This Project Covers

- Azure ML workspace setup, datastores, and data assets
- Training a CatBoost classifier on structured tabular data (Titanic dataset)
- Exporting the model to ONNX format for efficient deployment
- Environment management with Conda and Azure ML environments
- Model registration, versioning, and deployment using Managed Online Endpoints
- Real-time inference via REST API (score.py) using ONNX Runtime

---


## Installation

```bash
# Clone the repository
git clone https://github.com/kimiarezaei/Azure-ML-Model-Training-and-Deployment.git
cd Azure-ML-Model-Training-and-Deployment

# Setup environment
conda env create -f environments/conda_dependencies.yml
conda activate ml-env

# Train the model
python src/train.py

```


## Deploy the Model to Azure ML

- Register the model with Azure ML
- Deploy to a Managed Online Endpoint
- Use score.py for real-time inference

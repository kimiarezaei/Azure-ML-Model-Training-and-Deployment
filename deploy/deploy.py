from azure.ai.ml import MLClient, command
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Workspace, Data, Environment, Model, ManagedOnlineEndpoint, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# connect to azure account
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-ID",
    resource_group_name="azureml-kr",
    workspace_name="ml_ws"   # optional if you want to connect to an existing workspace
)

# make a workspace (if not exists)
workspace_name = "ml_ws"
ws_basic = Workspace(
    name=workspace_name,
    location="norwayeast",
    display_name="Basic ml_ws",
    description="Creating a workspace",
)
ml_client.workspaces.begin_create(ws_basic)

# Creat environment
environment_name = "ml-env1"
custom_env = Environment(
    name=environment_name,
    description="Environment for ML experiments",
    conda_file="./environments/conda_dependencies.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",  # CPU only image
    # image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04:latest"  # GPU enabled image
)
custom_job_env = ml_client.environments.create_or_update(custom_env)

#if environment already exist, get the environment:
for env in ml_client.environments.list():
    print(env.name, env.latest_version)

# list datastores
datastores = ml_client.datastores.list()
for ds in datastores:
    print(ds.name, "-", ds.type)

# create data asset
my_path = "./data/Titanic-Dataset.csv"
v1 = "initial"
data_asset_name = "TITANIC_DATASET"
my_data = Data(
    name=data_asset_name,
    version=v1,
    description="Titanic dataset for ML experiments",
    path=my_path,
    type=AssetTypes.URI_FILE,
)
ml_client.data.create_or_update(my_data)

#if data already exist, get the data:
data_asset = ml_client.data.get(name="TITANIC_DATASET", version=v1)

# Create and submit a command job
command_job = command(
    code="./src",
    command="python train.py --data_path ${{inputs.train_data}}",
    inputs={
        "train_data": "azureml:TITANIC_DATASET@latest"
    },
    environment="azureml:ml-env1@latest",
    compute="cpu-cluster",
    experiment_name="RF-titanicdata",
)
ml_client.jobs.create_or_update(command_job)

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="RF-endpoint",
    auth_mode="key"
    )
endpoint.traffic = {"RF-deployment": 100}
ml_client.online_endpoints.create_or_update(endpoint)

# Deploy model
deployment = ManagedOnlineDeployment(
    name="RF-deployment",
    endpoint_name="RF-endpoint",
    model="RF-model:1",
    environment="azureml:ml-env1@latest",
    code="./src",
    scoring_script="score.py",
    instance_type="Standard_DS3_v2",
    instance_count=1
)
ml_client.online_deployments.create_or_update(deployment)

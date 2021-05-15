# lightning-mlflow-pipeline  
Deeplearning pipeline for MNIST classification using  
pytorch + pytorch-lightning + omegaconf + mlflow 

## Docker Image
```shell
cd docker
docker build -t lightining-mlflow-pipeline:latest .
```

## Usage
MNIST classification by default
```shell
python main.py
```
Training detail is set by config.yaml  

main.py and dataset/dataset.py should be modified to train another dataset.

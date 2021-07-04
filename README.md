# lightning-mlflow-pipeline  
Deeplearning pipeline for MNIST classification using  
pytorch + pytorch-lightning + omegaconf + mlflow 


## Usage
MNIST classification by default
```shell
python main.py
```
Training detail is set by config.yaml  

main.py and dataset/dataset.py should be modified to train another dataset.

## Using Docker Image (example)
```shell
cd docker
docker build -t lightining-mlflow-pipeline:latest .
cd ..
docker run -it --rm -v $PWD:/workspace/ --shm-size 2g lightining-mlflow-pipeline:latest /bin/bash
python main.py
```

## Hyper-parameter search
you can search hyper-parameters using optuna.  
There should be some modification for parameter search.  
Default:
- python code: parameter_search.py
- configuration file: config_optuna.yaml
- db file: db/optuna.db

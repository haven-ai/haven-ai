# Quick Start with GCP 
The GCP scheduler involves the use of ```google ai-platform```, ```google cloud storage```, ```google container registry``` and ```docker```.

## 1. [Set up Google Cloud SDK and Docker](https://cloud.google.com/ai-platform/training/docs/custom-containers-training#before_you_begin)
Mark down the ```project id``` created in this step for future use.

## 2. [Set up Cloud Storage Bucket](https://cloud.google.com/ai-platform/training/docs/custom-containers-training#set-up-cloud-storage)
The bucket created in this step will be used to store the results and the dataset. Mark down the ```bucket name``` created and the ```region``` in this step for future use.

## 3. Modify the [job_config](https://github.com/haven-ai/haven-ai/blob/gcp/trainval.py#L87) for GCP
- container_hostname: The data center that the docker image will be pushed to. [Check all hostnames](https://cloud.google.com/container-registry/docs/pushing-and-pulling#add-registry).
- project_id: The project id created in step 1
- gcloud_savedir: The bucket name created in step 2
- region: The region used in step 2

## 4 Train the model
### 4.1 Train with CPU
```python
python trainval.py -e syn -sb results -r 1 -j gcp
```

### 4.2 Train with GPU
1. Rename the ```Dockerfile``` to ```Dockerfile_CPU```
2. Rename the ```Dockerfile_GPU``` to ```Dockerfile```
3. Add the following attributes to the [job_config](https://github.com/haven-ai/haven-ai/blob/gcp/trainval.py#L87)
    - scale-tire
    - master-machine-type
    - master-acceleration

    Check all [machine types](https://cloud.google.com/ai-platform/training/docs/using-gpus#compute-engine-machine-types-with-gpu) and the compatible accelerators supported by a [region](https://cloud.google.com/ai-platform/training/docs/regions#training_with_accelerators). 

4. Run 
```python 
python trainval.py -e syn -sb results -r 1 -j gcp
```

## 5. Visualize Results with Jupyter Notebook
Run the following command.
```python
jupyter notebook
```
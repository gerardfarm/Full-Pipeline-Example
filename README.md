# Full-Pipeline-Example
Implementation of a full pipeline example to train and evaluate a model on a specific dataset

### Usage
First, you need to prepare your docker registry. 
In `docker` folder, you could add all necessary libraries to Dockerfile along with all needed codes such as `functions.py`.

After building and pushing your docker image to your repository, you can now run your pipeline. The output is a .yaml file to be used in Kubeflow piepline.
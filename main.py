""" To run this pipeline, put into your terminal:
        dsl-compile --py main.py --output pipeline.yaml
"""

import kfp
import boto3

from train import Training
from eval import Evaluation
from preprocess import Preprocessing


# Define registry
AWS_REGION='us-east-1'
AWS_ACCOUNT_ID = boto3.client('sts').get_caller_identity().get('Account')

REPO_NAME = 'ali-repo'
BUCKET_NAME = 'ali-bucket-gerard'
DOCKER_REGISTRY = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(
                                                    AWS_ACCOUNT_ID, 
                                                    AWS_REGION, 
                                                    REPO_NAME
                                                )

# Create components
preprocess_task = kfp.components.create_component_from_func(Preprocessing, 
                                                    base_image=DOCKER_REGISTRY,
                                                  #  output_component_file = 'preprocessing.yaml'
                                                )

training_task = kfp.components.create_component_from_func(Training, 
                                                    base_image=DOCKER_REGISTRY,
                                                   # output_component_file = 'training.yaml'
                                                ) 

evaluation_task = kfp.components.create_component_from_func(Evaluation, 
                                                    base_image=DOCKER_REGISTRY,
                                                   # output_component_file = 'evaluation.yaml'
                                                )

# Create pipeline
@kfp.dsl.pipeline(
    name='Access S3', 
    description='A simple intro pipeline'
)
def pipeline_s3(bucket_name: str = BUCKET_NAME,
                data_path_in_s3: str = 'data/subset_images/',
                out_path: str = '/home/data/',
                AWS_REGION: str = 'us-east-1'
                ):

    # Upload your dataset to s3
    first_task = preprocess_task()

    # Download dataset, train your model and upload weights to s3
    second_task = training_task(first_task.outputs['feedback'], 
                                    bucket_name,
                                    data_path_in_s3,
                                    out_path,
                                    AWS_REGION)
    
    # Download weights, evaluate your model and upload results to s3
    third_task = evaluation_task(second_task.outputs['feedback'],
                                    bucket_name,
                                    data_path_in_s3,
                                    out_path,
                                    AWS_REGION)

if __name__ == "__main__":
    # execute only if run as a script
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline_s3,
        package_path='full-pipeline-example.yaml')
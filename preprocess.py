from typing import NamedTuple


# ================================================================
#                         Upload dataset to S3
# ================================================================
def Preprocessing() -> NamedTuple('My_Output',[('feedback', str)]):
    
    # You should upload your dataset to s3
    
    # import os
    # import boto3

    # conn_s3 = boto3.client('s3', region_name=AWS_REGION)
    
    # # Images names list
    # filenames = os.listdir(data_path)

    # # Upload all images to s3
    # for filename in filenames:
    #     conn_s3.upload_file(os.path.join(data_path, filename), 
    #                         bucket_name, 
    #                         os.path.join(output_path, filename))

    from collections import namedtuple
    feedback_msg = 'Done! Data are on S3.'
    func_output = namedtuple('MyOutput', ['feedback'])
    return func_output(feedback_msg)
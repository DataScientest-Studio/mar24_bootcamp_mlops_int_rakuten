import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv
import os
import tarfile

load_dotenv()

ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

def download_from_aws(s3_file, bucket, local_file_path, file_name=None):
    s3 = boto3.client('s3', 
                      aws_access_key_id=ACCESS_KEY, 
                      aws_secret_access_key=SECRET_KEY)
    if file_name is None:
        file_name = s3_file
    file_path = os.path.join(local_file_path, file_name)
    try:
        s3.download_file(bucket, s3_file, file_path)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def tar_folder(folder_path, tar_path=None):
    if tar_path is None:
        tar_path = folder_path +'.tar'
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    return tar_path




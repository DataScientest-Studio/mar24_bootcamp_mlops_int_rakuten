import tarfile
from dotenv import load_dotenv

load_dotenv()

def extract_tarfile(tar_path, file_path):
    with tarfile.open(tar_path) as tar:
        tar.extractall(file_path)#, filter='fully_trusted')
        
        


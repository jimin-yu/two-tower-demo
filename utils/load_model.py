import tensorflow as tf
import tarfile
import os

TEMP_DIR = 'temp'

class LoadModel:


  def __init__(self, s3_client, bucket_name):
    self.s3_client = s3_client
    self.bucket_name = bucket_name


  def call(self, model_name):
    if not os.path.exists(TEMP_DIR):
      os.makedirs(TEMP_DIR)

    s3_file_path = model_name + ".tar.gz"
    local_file_path = f'{TEMP_DIR}/{s3_file_path}'
    model_path = f'temp/{model_name}'

    self.__download_from_s3(s3_file_path, local_file_path)
    self.__unzip_tar_gz(local_file_path)
    
    loaded_model = self.__load_model(model_path)
    return loaded_model
  

  def __load_model(self, model_path):
    return tf.saved_model.load(model_path)


  def __unzip_tar_gz(self, tar_gz_file):
    with tarfile.open(tar_gz_file, "r:gz") as tar:
        tar.extractall(path=TEMP_DIR)
    os.remove(tar_gz_file)
  
  
  def __download_from_s3(self, s3_file_path, local_file_path):
    self.s3_client.download_file(self.bucket_name, s3_file_path, local_file_path)



import tensorflow as tf
import tarfile
import os
import shutil

TEMP_DIR = 'temp'

class SaveModel:
  def __init__(self, s3_client, bucket_name):
    self.s3_client = s3_client
    self.bucket_name = bucket_name
    self.model_version = 1


  def call(self, model, model_name, signatures=None):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    local_model_path = f'{TEMP_DIR}/{model_name}/{self.model_version}'

    self.__save_model_locally(model, local_model_path, signatures)
    tar_gz_file_name = self.__make_tar_gz(local_model_path)
    self.__upload_to_s3(model_name, tar_gz_file_name)

  
  def __save_model_locally(self, model, local_path, signatures):
    tf.saved_model.save(model, local_path, signatures=signatures)


  def __make_tar_gz(self, source_dir):
      tar_gz_file_name = source_dir + ".tar.gz"

      with tarfile.open(tar_gz_file_name, "w:gz") as tar:
          tar.add(source_dir, arcname=os.path.basename(source_dir))
          
      shutil.rmtree(source_dir)
      return tar_gz_file_name
  
  
  def __upload_to_s3(self, model_name, local_file_path):
     s3_file_path = os.path.join(model_name, os.path.basename(local_file_path))
     self.s3_client.upload_file(local_file_path, self.bucket_name, s3_file_path)
     os.remove(local_file_path)
     
     print(f"Uploaded {local_file_path} to s3://{self.bucket_name}/{s3_file_path}")




import os
import shutil
from google.cloud import storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "adhd-415110-5ffb2b65beea.json" 
def upload_folder_to_cloud_storage_client(bucket_name="booksdb", source_folder_path= "chroma_db", destination_folder_path="chroma_db"):
    """Uploads a folder to a GCS bucket, preserving directory structure.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_folder_path (str): The local path to the folder to upload.
        destination_folder_path (str): The destination path within the bucket 
                                      (can include subfolders).
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        for root, _, files in os.walk(source_folder_path):
            for file in files:
                local_path = os.path.join(root, file)
                remote_path = os.path.join(
                    destination_folder_path, 
                    local_path.replace(source_folder_path + os.sep, '')
                )
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_path)
        return "passed",1        
    except:
        return "Error occurred", 0            



def download_file_from_cloud_storage_client(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )
        return "passed",1        
    except:
        return "Error occurred", 0    




def download_folder_from_cloud_storage_client(bucket_name="booksdb", prefix= "chroma_db",destination_base_dir=""):
    """Downloads a blob from cloud storage, recreating directories locally."""
    try:
        if os.path.exists(prefix):
            # Directory exists, so delete it and its contents
            shutil.rmtree(prefix)
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        # Fetch blobs starting with "chroma_db"
        blobs = bucket.list_blobs(prefix=prefix)  
        # Print the names of the fetched blobs
        for blob in blobs:
            print(blob.name)
            # Split the blob name into directories and filename
            path_components = blob.name.split('\\')
            filename = path_components.pop()

            # Create the destination directory if it doesn't exist
            destination_dir = os.path.join(destination_base_dir, *path_components)
            os.makedirs(destination_dir, exist_ok=True)

            # Construct the full destination path
            destination_file_path = os.path.join(destination_dir, filename)

            # Download the blob
            blob.download_to_filename(destination_file_path)
            
            return "passed",1     
    except:
        return "Error occurred", 0    

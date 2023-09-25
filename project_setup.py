import logging
import os
import shutil
import tarfile

import boto3
import fire
import pandas as pd
import wget
from dotenv import load_dotenv


def main(force=False):
    logging.info("START")

    lfw_url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    download_dir = "./lfw_dataset"

    os.makedirs(download_dir, exist_ok=True)

    # download
    logging.info("Downloading LFW dataset...")
    lfw_tar_file = os.path.join(download_dir, "lfw.tgz")
    if not os.path.exists(lfw_tar_file) or force:
        if force and os.path.exists(lfw_tar_file):
            os.remove(lfw_tar_file)
            print(f"{lfw_tar_file} has been removed.")
        wget.download(lfw_url, out=lfw_tar_file)
        logging.info("\nDownload complete.")
    else:
        logging.info("LFW dataset already downloaded.")

    # extract
    logging.info("Extracting LFW dataset...")
    lfw_extracted_dir = os.path.join(download_dir, "lfw")
    if not os.path.exists(lfw_extracted_dir) or force:
        if force and os.path.exists(lfw_extracted_dir):
            shutil.rmtree(lfw_extracted_dir)
            print(f"{lfw_extracted_dir} has been removed.")
        with tarfile.open(lfw_tar_file, "r") as tar:
            tar.extractall(download_dir)
        logging.info("Extraction complete.")
    else:
        logging.info("LFW dataset already extracted.")

    # filter people with at least 2 images
    logging.info("Filtering out people with less than 2 images in the LFW dataset...")
    lfw_filtered_csv = os.path.join(download_dir, "lfw.csv")
    if not os.path.exists(lfw_filtered_csv) or force:
        if force and os.path.exists(lfw_filtered_csv):
            os.remove(lfw_filtered_csv)
            print(f"{lfw_filtered_csv} has been removed.")
        lfw_filtered_data = []
        for person_id, person in enumerate(os.listdir(lfw_extracted_dir)):
            person_path = os.path.join(lfw_extracted_dir, person)
            person_images = os.listdir(person_path)
            num_images = len(person_images)
            if num_images >= 2:
                for image_path in person_images:
                    lfw_filtered_data.append(
                        {
                            "person_id": person_id,
                            "person": person,
                            "image_path": image_path,
                        }
                    )
            else:
                shutil.rmtree(person_path)
        lfw_filtered_data = pd.DataFrame(lfw_filtered_data)
        lfw_filtered_data.to_csv(lfw_filtered_csv, index=False)
        logging.info("Filtering complete.")
    else:
        logging.info("LFW dataset already filtered.")

    # sync to s3
    logging.info("Syncing local LFW data to s3...")
    bucket_name = "facial-recognition-bucket"
    lfw_s3_prefix = "lfw_dataset/"
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        endpoint_url=os.environ["S3_URL"],
    )
    data_exists_in_s3 = "Contents" in s3.list_objects_v2(
        Bucket=bucket_name, Prefix=lfw_s3_prefix
    )
    if not data_exists_in_s3 or force:
        if data_exists_in_s3 and force:
            found_data = False
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=lfw_s3_prefix):
                if "Contents" in page:
                    s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={
                            "Objects": [{"Key": obj["Key"]} for obj in page["Contents"]]
                        },
                    )
                    found_data = True
            if found_data:
                logging.info(
                    f"All objects in prefix '{lfw_s3_prefix}' have been deleted."
                )
        for root, _, files in os.walk(download_dir):
            for file in files:
                local_path = os.path.join(root, file)
                s3_path = os.path.join(
                    lfw_s3_prefix, os.path.relpath(local_path, download_dir)
                )
                s3.upload_file(local_path, bucket_name, s3_path)
        logging.info("Syncing complete.")
    else:
        logging.info("LFW dataset already in s3.")


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")  # take environment variables from .env
    logging.basicConfig(level=os.environ["LOGGING_LEVEL"])
    fire.Fire(main)

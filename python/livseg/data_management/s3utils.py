"""Manages the S3 bucket access"""
import os

import boto3 as bt
from tqdm import tqdm
class S3UpLoader:
    def __init__(
        self,
        bucketconfig: dict,
        session,
        instance="dl1",
        log_dir="./performance_logs",
        tensorboard_log_dir="./logs",
        history_dir="./history",
        weights_dir="./weights",
    ) -> None:
        instances = ["dl1", "p4d", "p3dn"]
        if instance not in instances:
            raise f"instance arg must be in {instances}"
        self.tensorboard_logs_dir = os.path.abspath(tensorboard_log_dir)
        self.log_dir = os.path.abspath(log_dir)
        self.instance = instance
        self.history_dir = history_dir
        self.weights_dir = weights_dir
        self.session = session
        bucketname = bucketconfig.pop("bucketname")
        self.bucket = bt.resource(
                **bucketconfig
            ).Bucket(bucketname)

    def upload_history(self):
        filename = f"{self.session}.json"
        object = self.bucket.Object(f"{self.instance}/history/{filename}")
        object.upload_file(os.path.join(self.history_dir, filename))

    def upload_tensorboard_logs(self):
        path = os.path.join(self.tensorboard_logs_dir, self.session)
        assert os.path.exists(path), f"{path} does not exist"
        for main, _, files in os.walk(path):
            if files:
                for file in files:
                    dirs = main.split("/")
                    index_of_session = dirs.index(self.session)
                    self.bucket.Object(
                        f'{self.instance}/tensorboard_logs/{"/".join(dirs[index_of_session:])}/{file}'
                    ).upload_file(os.path.join(main, file))

    def upload_logs(self):
        path = os.path.join(self.log_dir, self.session)
        for file in os.listdir(path):
            self.bucket.Object(f"{self.instance}/logs/{self.session}/{file}").upload_file(
                os.path.join(path, file)
            )

    def upload_weights(self):
        filename = f"{self.session}.h5"
        object = self.bucket.Object(f"{self.instance}/weights/{filename}")
        object.upload_file(os.path.join(self.weights_dir, filename))


def download_data(bucketconfig, prefix="DATA/original_data/", target=""):
    bucketname = bucketconfig.pop("bucketname")
    bucket = bt.resource(
                **bucketconfig
            ).Bucket(bucketname)
    if not os.path.exists(target):
        os.makedirs(target)
    for my_bucket_object in tqdm(list(bucket.objects.filter(Prefix=prefix))):
        suffix = my_bucket_object.key.split("/")[-1]
        if suffix:
            my_bucket_object.Object().download_file(os.path.join(target, suffix))

"""Perform a training model task"""

import argparse
import json
import os
import sys
from datetime import datetime
from random import seed

now = datetime.now()

import logging

import numpy as np

project_path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
python_module_path = os.path.join(project_path, "python")
sys.path.append(os.path.abspath(python_module_path))

parser = argparse.ArgumentParser()

parser.add_argument(
    "-i",
    "--input_dir",
    dest="input_dir",
    type=str,
    help="Data directory",
    required=True,
)
parser.add_argument(
    "-bs", "--batch_size", dest="batch_size", default=1, type=int, help="Batch size"
)

parser.add_argument(
    "-e", "--epochs", dest="epochs", default=1, type=int, help="Number of epochs"
)

parser.add_argument(
    "-instance",
    dest="instance",
    type=str,
    choices=["dl1", "p4d", "p3dn"],
    help="Type of instance where the script is launched",
    required=True,
)

parser.add_argument(
    "--mixed_precision",
    dest="mixed_precision",
    action="store_true",
    help="Enables mixed_precision",
)

parser.add_argument(
    "-l",
    "--logdir",
    dest="logdir",
    default="./logs",
    type=str,
    help="Tensorboard logs directory",
)

args = parser.parse_args()

use_gaudi = args.instance == "dl1"

with open(
    os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "bucket_config.json")),
    "r",
) as f:
    bucketconfig = json.load(f)
    all(bucketconfig[k] != "" for k in ["service_name", "region_name", "bucketname"])


if use_gaudi:
    try:
        from habana_frameworks.tensorflow import load_habana_module

        load_habana_module()
    except:
        raise ImportError("Habana import error")

horovod = True
try:
    import horovod.tensorflow.keras as hvd

    hvd.init()
except Exception:
    horovod = False

import tensorflow as tf
from livseg.data_management.data_augmenter import DataAugmenter
from livseg.data_management.dataset_handler import DatasetHandler, DatasetPartitioner
from livseg.data_management.loader import X_COL, Y_COL
from livseg.data_management.s3utils import S3UpLoader
from livseg.model.callbacks import BenchmarkClbck
from livseg.model.metrics import dice_coef
from livseg.model.unet2d import unet2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam

if args.mixed_precision:
    tf.keras.mixed_precision.set_global_policy(
        f'mixed_{"b" if use_gaudi else ""}float16'
    )

if not use_gaudi:
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank() if horovod else 0], "GPU"
        )


np.random.seed(42)

tmp_dir = args.input_dir
batch_size = args.batch_size
epochs = args.epochs
log_dir = args.logdir

hvd_size = hvd.size() if horovod else 1
hvd_rank = hvd.rank() if horovod else 0

session = (
    f"session_{batch_size}_{'mixed_precision' if args.mixed_precision else 'float32'}"
)


try:
    os.makedirs(os.path.join("performance_logs", session))
except:
    pass


log_Format = "%(asctime)s - %(levelname)s : %(message)s"

logging.basicConfig(
    filename=os.path.join("performance_logs", session, f"{hvd_rank}.log"),
    filemode="w",
    format=log_Format,
    level=logging.INFO,
)

logging.info(f"Starting {session}")

directory_handler = DatasetHandler(args.input_dir)
full_df_train, full_df_test = directory_handler.get_train_test_split_from_ids()

train_splitter = DatasetPartitioner(full_df_train, hvd_size)
test_splitter = DatasetPartitioner(full_df_test, hvd_size)

df_train = train_splitter.get_partition(hvd_rank)
df_test = test_splitter.get_partition(hvd_rank)


train_ds = tf.data.Dataset.from_tensor_slices(
    (df_train[X_COL].tolist(), df_train[Y_COL].tolist())
)
test_ds = tf.data.Dataset.from_tensor_slices(
    (df_test[X_COL].tolist(), df_test[Y_COL].tolist())
)

train_dataset_size = len(train_ds)
test_dataset_size = len(test_ds)

AUTOTUNE = tf.data.AUTOTUNE
DTYPES = [tf.float32, tf.uint8]


def map_func(feature_path, label_path):
    feature = np.load(feature_path)
    label = np.load(label_path).astype(np.uint8)
    return feature, label


data_reader_fn = lambda item1, item2: tf.numpy_function(
    map_func, [item1, item2], DTYPES
)

map_kwargs = dict(map_func=data_reader_fn, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.map(**map_kwargs)
test_ds = test_ds.map(**map_kwargs)


def prefetch(dataset):
    """Dataset prefetching function"""
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    if use_gaudi:
        dataset.apply(
            tf.data.experimental.prefetch_to_device(
                device="/device:HPU:0", buffer_size=batch_size * hvd.size()
            )
        )
    return dataset


def data_augmentation(x, y):
    data_augmenter = DataAugmenter()
    return (data_augmenter.transform(x), data_augmenter.transform(y))


def configure_for_performance(ds: tf.data.Dataset):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=train_dataset_size)
    ds = ds.repeat()
    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            map_func=data_augmentation,
            num_parallel_calls=AUTOTUNE,
            batch_size=batch_size,
            drop_remainder=True,
        )
    )
    ds = prefetch(ds)
    return ds


train_ds = configure_for_performance(train_ds)
test_ds = configure_for_performance(test_ds)

model = unet2D(input_shape=(512, 512, 1))

optimizer = Adam(1e-4 * hvd_size)
if horovod:
    optimizer = hvd.DistributedOptimizer(optimizer)


if hvd_rank == 0:
    model.summary(line_length=120)

model.compile(
    optimizer=optimizer,
    loss=BinaryCrossentropy(),
    metrics=[dice_coef],
)


tensorboard_callback = TensorBoard(
    log_dir=os.path.join(log_dir, session, str(hvd_rank)),
    update_freq="epoch",
)

callbacks = [
    tensorboard_callback,
    BenchmarkClbck(batch_size, hvd_size),
]

if horovod:
    callbacks.append(
        [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ]
    )

history = model.fit(
    train_ds,
    # validation_data=test_ds,
    steps_per_epoch=train_dataset_size / batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=int(hvd_rank == 0),
    # validation_freq=2,
)

end_time = datetime.now() - now
logging.info(f"GLOBAL TRAINING TIME:{end_time.total_seconds()} seconds")

if hvd_rank == 0:

    weights_dir = "./weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    save_model(model, os.path.join(weights_dir, f"{session}.h5"))

    history_path = "./history"
    if not os.path.exists(history_path):
        os.mkdir(history_path)

    for k, v in history.history.items():
        history.history[k] = list(map(lambda x: float(x), v))

    results = {
        "history": history.history,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    with open(os.path.join(history_path, f"{session}.json"), "w") as f:
        json.dump(results, f)
        f.close()

    uploader = S3UpLoader(
        bucketconfig=bucketconfig,
        session=session,
        instance=args.instance,
        tensorboard_log_dir=log_dir,
        history_dir=history_path,
        weights_dir=weights_dir,
    )

    uploader.upload_history()
    uploader.upload_tensorboard_logs()
    uploader.upload_logs()
    uploader.upload_weights()

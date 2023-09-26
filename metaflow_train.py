import logging
import os

from dotenv import load_dotenv
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow.cards import Image


class PrintLogger(logging.Logger):
    """
    Metaflow uses python `print()` to log stuff. I just don't like it very much.
    """

    def handle(self, record):
        print(f"{record.name}: {record.msg}")


class FacialRecognitionTrainFlow(FlowSpec):
    bucket = Parameter("bucket", default="facial-recognition-bucket", type=str)
    dataset_root_key = Parameter(
        "dataset_root_key",
        default="lfw_dataset",
        type=str,
    )
    dataset_csv_key = Parameter(
        "dataset_csv",
        default="lfw_dataset/lfw.csv",
        type=str,
    )
    dataset_images_root = Parameter(
        "dataset_filename",
        default="lfw_dataset/lfw/",
        type=str,
    )
    dataset_in_batch_num_samples_per_label = Parameter(
        "dataset_in_batch_num_samples_per_label", default=2, type=int
    )
    train_people_frac = Parameter(
        "train_people_frac",
        default=0.6,
        type=float,
    )
    dev_people_frac = Parameter(
        "dev_people_frac",
        default=0.2,
        type=float,
    )
    test_people_frac = Parameter(
        "test_people_frac",
        default=0.2,
        type=float,
    )
    model_name = Parameter(
        "model_name", default="tf_efficientnet_b0.ns_jft_in1k", type=str
    )
    model_init_kwargs = Parameter(
        "model_init_kwargs",
        default={"src_embeddings_dim": 1280, "tgt_embeddings_dim": 300},
        type=dict,
    )
    optimizer = Parameter("optimizer", default="adamw", type=str)
    loss = Parameter("loss", default="batch-hard-soft-margin", type=str)
    distance = Parameter("distance", default="euclidean", type=str)
    learning_rate = Parameter(
        "learning_rate",
        default=3e-4,
        type=float,
    )
    weight_decay = Parameter(
        "weight_decay",
        default=1e-2,
        type=float,
    )
    max_epochs = Parameter("max_epochs", default=1, type=int)
    evals_per_epoch = Parameter("evals_per_epoch", default=1, type=int)
    early_stopping = Parameter("early_stopping", default=False, type=bool)
    early_stopping_patience = Parameter("early_stopping_patience", default=1, type=int)
    train_batch_size = Parameter("train_batch_size", default=64, type=int)
    eval_batch_size = Parameter("eval_batch_size", default=64, type=int)
    resize_hw = Parameter("resize_hw", default=(224, 224), type=tuple)
    norm_mean = Parameter("norm_mean", default=(0.485, 0.456, 0.406), type=tuple)
    norm_std = Parameter("norm_std", default=(0.229, 0.224, 0.225), type=tuple)
    augmentation_level = Parameter("augmentation_level", default="high", type=str)
    clip_grad_norm = Parameter("clip_grad_norm", default=True, type=bool)

    @property
    def logger(self):
        """
        Get the logger for this class
        """
        logger = PrintLogger(name=self.__class__.__name__, level=logging.INFO)
        return logger

    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.

        """
        self.logger.info("FacialRecognitionTrainFlow is starting.")
        self.next(self.validate_params)

    @step
    def validate_params(self):
        """
        Verifies that user informed `Parameters` are within expected ranges and values.

        """
        assert (
            self.train_people_frac + self.dev_people_frac + self.test_people_frac
        ) == 1, "train, dev, test people frac should sum to one"
        assert (
            self.dataset_in_batch_num_samples_per_label > 1
        ), "`dataset_in_batch_num_samples_per_label` should be > 1"
        assert self.model_name in [
            "resnet18",
            "efficientnet_b0",
            "tf_efficientnet_b0.ns_jft_in1k",
        ], "`model_name` not implemented"
        assert self.optimizer in ["sgd", "adamw"], "`optimizer` not implemented"
        assert self.optimizer in ["sgd", "adamw"], "`optimizer` not implemented"
        assert self.loss in [
            "batch-hard-soft-margin",
            "batch-hard",
        ], "`loss` not implemented"
        assert self.distance in ["euclidean", "cosine"], "`distance` not implemented"
        assert (
            1e-2 >= self.learning_rate and self.learning_rate >= 1e-5
        ), "`learning_rate` too big or too small"
        assert (
            1e-1 >= self.weight_decay and self.weight_decay >= 0
        ), "`weight_decay` too big or too small"
        assert (
            self.train_batch_size >= 16
        ), "`train_batch_size` cant be that small, we need to sample triplets online"
        assert self.max_epochs >= 1, "`max_epochs`should be positive"
        assert self.evals_per_epoch >= 1, "`evals_per_epoch` should be positive"
        if self.early_stopping:
            assert self.early_stopping_patience < (
                self.max_epochs * self.evals_per_epoch
            ), "`early_stopping_patience` should be smaller than the total number of evaluations"
        assert (
            self.resize_hw[0] > 1 and self.resize_hw[1] > 1
        ), "`resize_hw` should be positive"
        assert (
            len(self.resize_hw) == 2
        ), "`resize_hw` should be a tuple of height and width"
        assert (
            len(self.norm_mean) == 3
        ), "`norm_mean` should be a tuple of channel means (R, G and B)"
        assert (
            len(self.norm_std) == 3
        ), "`norm_std` should be a tuple of channel stds (R, G and B)"
        assert self.augmentation_level in [
            "none",
            "low",
            "high",
        ], "`augmentation_level` not implemented"
        self.logger.info("Params OK.")
        self.next(self.train_dev_test_split)

    @step
    def train_dev_test_split(self):
        """
        Splits data into train, dev and test sets, mining static hard triplets for the last two.
        Saves splits at a unique path on s3.

        """
        import json
        import os
        from uuid import uuid4

        import boto3
        import numpy as np
        import pandas as pd
        from supertriplets.encoder import PretrainedSampleEncoder
        from supertriplets.evaluate import HardTripletsMiner
        from supertriplets.sample import ImageSample

        from src.train.utils import (
            delete_path,
            s3_download_all,
            s3_load_csv,
            s3_save_csv,
        )

        # reading csv data
        self.logger.info("Loading csv from s3..")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            endpoint_url=os.environ["S3_URL"],
        )
        df = s3_load_csv(
            s3_client=s3_client, src_bucket=self.bucket, src_key=self.dataset_csv_key
        )
        self.logger.info("Downloading dataset from s3..")
        saving_folder_name = os.path.join("/tmp", str(uuid4()))
        s3_download_all(
            s3_client=s3_client,
            src_bucket=self.bucket,
            src_key=self.dataset_images_root,
            tgt_local_directory=saving_folder_name,
        )

        # wrangle columns to supertriplets expected format
        self.logger.info("Wrangling data to expected format..")
        df["label"] = df["person_id"].astype(int)
        df["image_path"] = (df["person"] + "/" + df["image_path"]).apply(
            lambda x: os.path.join(saving_folder_name, x)
        )
        df = df[["label", "image_path"]]

        # split people (labels) into train, dev, test data
        self.logger.info("Splitting people (labels) into train, dev, test..")
        available_ids = df.label.unique().tolist()
        np.random.seed(42)
        np.random.shuffle(available_ids)

        num_ids = len(available_ids)
        train_size = int(self.train_people_frac * num_ids)
        dev_size = int(self.dev_people_frac * num_ids)
        test_size = num_ids - train_size - dev_size
        self.metadata = {
            "trainset_num_people": train_size,
            "devset_num_people": dev_size,
            "testset_num_people": test_size,
        }

        train_set = df[df.label.isin(available_ids[:train_size])].reset_index(drop=True)
        dev_set = df[
            df.label.isin(available_ids[train_size : train_size + dev_size])
        ].reset_index(drop=True)
        test_set = df[
            df.label.isin(available_ids[train_size + dev_size :])
        ].reset_index(drop=True)

        # finding dev and test static triplets
        self.logger.info("Mining hard triplets for the dev and test dataset..")
        device = "cuda:0"
        dev_examples = [
            ImageSample(image_path=image_path, label=label)
            for image_path, label in zip(dev_set["image_path"], dev_set["label"])
        ]
        test_examples = [
            ImageSample(image_path=image_path, label=label)
            for image_path, label in zip(test_set["image_path"], test_set["label"])
        ]

        pretrained_encoder = PretrainedSampleEncoder(modality="image")
        dev_embeddings = pretrained_encoder.encode(
            examples=dev_examples, device=device, batch_size=self.eval_batch_size
        )
        test_embeddings = pretrained_encoder.encode(
            examples=test_examples, device=device, batch_size=self.eval_batch_size
        )
        del pretrained_encoder

        hard_triplet_miner = HardTripletsMiner(use_gpu_powered_index_if_available=True)
        (
            dev_anchor_examples,
            dev_positive_examples,
            dev_negative_examples,
        ) = hard_triplet_miner.mine(
            examples=dev_examples,
            embeddings=dev_embeddings,
            normalize_l2=True,
            sample_from_topk_hardest=5,
        )
        (
            test_anchor_examples,
            test_positive_examples,
            test_negative_examples,
        ) = hard_triplet_miner.mine(
            examples=test_examples,
            embeddings=test_embeddings,
            normalize_l2=True,
            sample_from_topk_hardest=5,
        )
        del hard_triplet_miner

        dev_set = pd.DataFrame(
            [
                {
                    **{"anchor_" + k: v for k, v in a.data().items()},
                    **{"positive_" + k: v for k, v in p.data().items()},
                    **{"negative_" + k: v for k, v in n.data().items()},
                }
                for a, p, n in zip(
                    dev_anchor_examples, dev_positive_examples, dev_negative_examples
                )
            ]
        )
        test_set = pd.DataFrame(
            [
                {
                    **{"anchor_" + k: v for k, v in a.data().items()},
                    **{"positive_" + k: v for k, v in p.data().items()},
                    **{"negative_" + k: v for k, v in n.data().items()},
                }
                for a, p, n in zip(
                    test_anchor_examples, test_positive_examples, test_negative_examples
                )
            ]
        )

        # fix image_path col
        train_set["image_path"] = train_set["image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        dev_set["anchor_image_path"] = dev_set["anchor_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        dev_set["positive_image_path"] = dev_set["positive_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        dev_set["negative_image_path"] = dev_set["negative_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        test_set["anchor_image_path"] = test_set["anchor_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        test_set["positive_image_path"] = test_set["positive_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )
        test_set["negative_image_path"] = test_set["negative_image_path"].apply(
            lambda x: x.split(saving_folder_name)[-1][1:]
        )

        # saving train, dev and test data
        self.logger.info(
            "Saving training dataset and dev/test mined hard triplets datasets.."
        )
        random_id = str(uuid4())
        self.train_set_key = f"/train/{random_id}/train.csv"
        s3_save_csv(
            df=train_set,
            s3_client=s3_client,
            target_bucket=self.bucket,
            target_key=self.train_set_key,
        )
        self.dev_set_key = f"/train/{random_id}/dev.csv"
        s3_save_csv(
            df=dev_set,
            s3_client=s3_client,
            target_bucket=self.bucket,
            target_key=self.dev_set_key,
        )
        self.test_set_key = f"/train/{random_id}/test.csv"
        s3_save_csv(
            df=test_set,
            s3_client=s3_client,
            target_bucket=self.bucket,
            target_key=self.test_set_key,
        )

        self.metadata.update(
            {
                "trainset_size": len(train_set),
                "devset_size": len(dev_set),
                "testset_size": len(test_set),
            }
        )
        self.logger.info(json.dumps(self.metadata, indent=4))

        # cleanup
        self.logger.info("Deleting everything downloaded..")
        delete_path(saving_folder_name)
        self.next(self.finetune)

    @card(type="blank")
    @step
    def finetune(self):
        """
        Loads data splits from s3 and fine-tunes a pretrained neural network for metric learning with online mined hard triplets.


        """
        import json
        import math
        import os
        from io import BytesIO
        from uuid import uuid4

        import boto3
        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from sklearn.calibration import calibration_curve
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import accuracy_score, f1_score
        from supertriplets.dataset import OnlineTripletsDataset, StaticTripletsDataset
        from supertriplets.distance import CosineDistance, EuclideanDistance
        from supertriplets.evaluate import TripletEmbeddingsEvaluator
        from supertriplets.loss import (
            BatchHardSoftMarginTripletLoss,
            BatchHardTripletLoss,
        )
        from supertriplets.sample import ImageSample
        from supertriplets.utils import move_tensors_to_device
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        from src.inference.config import ProductionConfig
        from src.inference.model import init_model, load_state_dict, save_state_dict
        from src.train.encoding import get_triplet_embeddings
        from src.train.preprocessing import get_augmentations, load_input_example
        from src.train.utils import (
            MetricTracker,
            delete_path,
            get_cosine_similarity_scores_shuffled,
            s3_download_all,
            s3_load_csv,
            set_seed,
        )

        # reading csv data
        self.logger.info("Downloading dataset from s3..")
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            endpoint_url=os.environ["S3_URL"],
        )
        saving_folder_name = os.path.join("/tmp", str(uuid4()))
        s3_download_all(
            s3_client=s3_client,
            src_bucket=self.bucket,
            src_key=self.dataset_images_root,
            tgt_local_directory=saving_folder_name,
        )

        # locking random seeds
        self.logger.info("Locking random seeds..")
        set_seed(42)

        # loading train, dev and test data
        self.logger.info(
            "Loading training dataset and dev/test mined hard triplets datasets.."
        )
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            endpoint_url=os.environ["S3_URL"],
        )
        train_data = s3_load_csv(
            s3_client=s3_client, src_bucket=self.bucket, src_key=self.train_set_key
        )
        dev_data = s3_load_csv(
            s3_client=s3_client, src_bucket=self.bucket, src_key=self.dev_set_key
        )
        test_data = s3_load_csv(
            s3_client=s3_client, src_bucket=self.bucket, src_key=self.test_set_key
        )

        # creating supertriplet samples
        self.logger.info("Creating Samples from dfs..")

        # train ImageSamples
        train_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(train_data["image_path"], train_data["label"])
        ]

        # dev ImageSamples
        dev_anchor_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                dev_data["anchor_image_path"], dev_data["anchor_label"]
            )
        ]
        dev_positive_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                dev_data["positive_image_path"], dev_data["positive_label"]
            )
        ]
        dev_negative_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                dev_data["negative_image_path"], dev_data["negative_label"]
            )
        ]

        # test ImageSamples
        test_anchor_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                test_data["anchor_image_path"], test_data["anchor_label"]
            )
        ]
        test_positive_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                test_data["positive_image_path"], test_data["positive_label"]
            )
        ]
        test_negative_examples = [
            ImageSample(
                image_path=os.path.join(saving_folder_name, image_path), label=label
            )
            for image_path, label in zip(
                test_data["negative_image_path"], test_data["negative_label"]
            )
        ]

        # loading model
        self.logger.info("Loading model..")
        device = "cuda:0"

        model = init_model(
            model_name=self.model_name, model_init_kwargs=self.model_init_kwargs
        )
        model.to(device)
        model.eval()

        # defining torch datasets
        self.logger.info("Building torch train/dev/test Datasets..")
        trainset = OnlineTripletsDataset(
            examples=train_examples,
            in_batch_num_samples_per_label=self.dataset_in_batch_num_samples_per_label,
            batch_size=self.train_batch_size,
            sample_loading_func=load_input_example,
            sample_loading_kwargs={
                "transform": get_augmentations(
                    level=self.augmentation_level,
                    resize_hw=self.resize_hw,
                    norm_mean=self.norm_mean,
                    norm_std=self.norm_std,
                )
            },
        )
        devset = StaticTripletsDataset(
            anchor_examples=dev_anchor_examples,
            positive_examples=dev_positive_examples,
            negative_examples=dev_negative_examples,
            sample_loading_func=load_input_example,
            sample_loading_kwargs={
                "transform": get_augmentations(
                    level="none",
                    resize_hw=self.resize_hw,
                    norm_mean=self.norm_mean,
                    norm_std=self.norm_std,
                )
            },
        )
        testset = StaticTripletsDataset(
            anchor_examples=test_anchor_examples,
            positive_examples=test_positive_examples,
            negative_examples=test_negative_examples,
            sample_loading_func=load_input_example,
            sample_loading_kwargs={
                "transform": get_augmentations(
                    level="none",
                    resize_hw=self.resize_hw,
                    norm_mean=self.norm_mean,
                    norm_std=self.norm_std,
                )
            },
        )

        # defining torch dataloaders
        self.logger.info("Building torch train/dev/test DataLoaders..")
        trainloader = DataLoader(
            dataset=trainset,
            batch_size=self.train_batch_size,
            num_workers=1,
            drop_last=True,
        )
        devloader = DataLoader(
            dataset=devset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )
        testloader = DataLoader(
            dataset=testset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )

        # training config
        self.logger.info("Configuring criterion, optimizer..")
        param_optimizer = model.parameters()

        match self.optimizer:
            case "sgd":
                optimizer = torch.optim.SGD(
                    param_optimizer,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    param_optimizer,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise NotImplementedError(f"Optimizer `{self.optimizer}` not available")
        match self.loss, self.distance:
            case "batch-hard", "cosine":
                criterion = BatchHardTripletLoss(
                    distance=CosineDistance(alredy_l2_normalized_vectors=False),
                    margin=5,
                )
            case "batch-hard", "euclidean":
                criterion = BatchHardTripletLoss(
                    distance=EuclideanDistance(squared=False), margin=5
                )
            case "batch-hard-soft-margin", "cosine":
                criterion = BatchHardSoftMarginTripletLoss(
                    distance=CosineDistance(alredy_l2_normalized_vectors=False)
                )
            case "batch-hard-soft-margin", "euclidean":
                criterion = BatchHardSoftMarginTripletLoss(
                    distance=EuclideanDistance(squared=False)
                )
            case _:
                raise NotImplementedError(
                    f"Loss `{self.loss}` with distance `{self.distance}` not available"
                )

        # init embeddings evaluator
        triplet_embeddings_evaluator = TripletEmbeddingsEvaluator(
            calculate_by_cosine=True,
            calculate_by_manhattan=True,
            calculate_by_euclidean=True,
        )

        # calculate initial metrics
        self.logger.info("Calculating baseline dev accuracy..")
        dev_triplet_embeddings = get_triplet_embeddings(
            dataloader=devloader, model=model, device=device
        )
        dev_start_accuracies = triplet_embeddings_evaluator.evaluate(
            embeddings_anchors=dev_triplet_embeddings["anchors"],
            embeddings_positives=dev_triplet_embeddings["positives"],
            embeddings_negatives=dev_triplet_embeddings["negatives"],
        )
        self.logger.info(json.dumps(dev_start_accuracies, indent=4))
        metric_tracker = MetricTracker()
        for k, v in dev_start_accuracies.items():
            metric_tracker.log(name=f"dev_{k}", value=v, epoch=1, step=0)

        # optimization loop
        self.logger.info("Training..")
        step = 0
        max_step = self.max_epochs * len(trainloader)
        eval_steps = [
            math.ceil(len(trainloader) / (self.evals_per_epoch))
            * i  # within epoch steps case
            if i % self.evals_per_epoch != 0
            else (i / self.evals_per_epoch)
            * len(trainloader)  # epoch ending steps case
            for i in range(1, (self.evals_per_epoch * self.max_epochs) + 1)
        ]
        this_path_id = str(uuid4())
        out_path = os.path.join("/tmp", this_path_id)
        os.makedirs(out_path, exist_ok=True)
        state_dict_path = os.path.join(out_path, "facial_recognition_model.pth")
        curr_es_patience = 0
        best_accuracy = -999
        all_train_losses = []
        for epoch in range(1, self.max_epochs + 1):
            self.logger.info(f"Start of epoch {epoch}/{self.max_epochs}")
            for batch in tqdm(
                trainloader, total=len(trainloader), desc=f"Epoch {epoch}"
            ):
                model.train()
                data = batch["samples"]
                labels = move_tensors_to_device(obj=data.pop("label"), device=device)
                inputs = move_tensors_to_device(obj=data, device=device)

                optimizer.zero_grad()

                embeddings = model(**inputs)
                loss = criterion(embeddings=embeddings, labels=labels)
                loss.backward()
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1, norm_type=2
                    )

                optimizer.step()
                step += 1

                all_train_losses.append(loss.item())
                metric_tracker.log(
                    name="train_loss_50batches_movavg",
                    value=np.mean(all_train_losses[-50:]),
                    epoch=epoch,
                    step=step,
                )
                metric_tracker.log(
                    name="train_loss", value=loss.item(), epoch=epoch, step=step
                )

                if step in eval_steps:
                    with torch.no_grad():
                        self.logger.info(
                            f"Step {int(step)}/{int(max_step)}: evaluating.."
                        )
                        dev_triplet_embeddings = get_triplet_embeddings(
                            dataloader=devloader, model=model, device=device
                        )
                        dev_accuracies = triplet_embeddings_evaluator.evaluate(
                            embeddings_anchors=dev_triplet_embeddings["anchors"],
                            embeddings_positives=dev_triplet_embeddings["positives"],
                            embeddings_negatives=dev_triplet_embeddings["negatives"],
                        )
                        for k, v in dev_accuracies.items():
                            metric_tracker.log(
                                name=f"dev_{k}", value=v, epoch=epoch, step=step
                            )

                        dev_max_accuracy = max(dev_accuracies.values())

                        if dev_max_accuracy >= best_accuracy:
                            self.logger.info(
                                f"New best model: dev_max_accuracy {round(best_accuracy, 3)} -> {round(dev_max_accuracy, 3)}"
                            )
                            save_state_dict(
                                model=model, state_dict_path=state_dict_path
                            )
                            best_accuracy = dev_max_accuracy
                            if self.early_stopping:
                                curr_es_patience = 0
                        else:
                            self.logger.info("No increase in accuracy")
                            if self.early_stopping:
                                curr_es_patience += 1
                                self.logger.info(f"ES patience: {curr_es_patience}")
                                if curr_es_patience >= self.early_stopping_patience:
                                    self.logger.info(f"ES max patience reached")
                                    break
            else:
                continue
            break
        self.logger.info("Ending of optimization loop")

        # loading and encoding dev/test for score calibration
        self.logger.info("Loading best model")
        model = init_model(
            model_name=self.model_name, model_init_kwargs=self.model_init_kwargs
        )
        model = load_state_dict(model=model, state_dict_path=state_dict_path)

        self.logger.info("Encoding dev triplets")
        dev_triplet_embeddings = get_triplet_embeddings(
            dataloader=devloader, model=model, device=device
        )
        self.logger.info("Encoding test triplets")
        test_triplet_embeddings = get_triplet_embeddings(
            dataloader=testloader, model=model, device=device
        )
        self.logger.info(
            "Creating scores dataframes for dev/test cos_sim(anchor embeddings, pos/neg embeddings)"
        )
        dev_scores = get_cosine_similarity_scores_shuffled(
            anchor_embeddings=dev_triplet_embeddings["anchors"],
            positive_embeddings=dev_triplet_embeddings["positives"],
            negative_embeddings=dev_triplet_embeddings["negatives"],
        )
        test_scores = get_cosine_similarity_scores_shuffled(
            anchor_embeddings=test_triplet_embeddings["anchors"],
            positive_embeddings=test_triplet_embeddings["positives"],
            negative_embeddings=test_triplet_embeddings["negatives"],
        )

        # fitting calibrator
        self.logger.info("Fitting calibration model on dev set")
        calibrator = IsotonicRegression(
            y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
        )
        calibrator.fit(X=dev_scores["score"].values, y=dev_scores["match"].values)

        # calibrating
        self.logger.info("Calibrating test set scores")
        test_scores["calib_score"] = calibrator.transform(test_scores["score"].values)

        # measures
        self.logger.info("Calculating test set metrics")
        metric_tracker.log(
            name=f"test_calibrated_accuracy",
            value=accuracy_score(
                y_true=test_scores["match"].values,
                y_pred=(test_scores["calib_score"] >= 0.5).astype(int),
            ),
            epoch=epoch,
            step=step,
        )
        metric_tracker.log(
            name=f"test_calibrated_f1_score",
            value=f1_score(
                y_true=test_scores["match"].values,
                y_pred=(test_scores["calib_score"] >= 0.5).astype(int),
            ),
            epoch=epoch,
            step=step,
        )

        test_accuracies = triplet_embeddings_evaluator.evaluate(
            embeddings_anchors=test_triplet_embeddings["anchors"],
            embeddings_positives=test_triplet_embeddings["positives"],
            embeddings_negatives=test_triplet_embeddings["negatives"],
        )
        for k, v in test_accuracies.items():
            metric_tracker.log(name=f"test_{k}", value=v, epoch=epoch, step=step)

        # metrics
        self.logger.info("Saving metrics")
        self.metrics = metric_tracker.metrics

        # cards
        self.logger.info("Saving plots")

        self.logger.info("Saving reliability diagram..")
        y_true = test_scores["match"].values
        yhat_uncalibrated = test_scores["score"].values
        yhat_uncalibrated = np.clip(yhat_uncalibrated, a_min=0, a_max=1)
        yhat_calibrated = test_scores["calib_score"].values
        fop_uncalibrated, mpv_uncalibrated = calibration_curve(
            y_true=y_true,
            y_prob=yhat_uncalibrated,
            n_bins=10,
        )
        fop_calibrated, mpv_calibrated = calibration_curve(
            y_true=y_true, y_prob=yhat_calibrated, n_bins=10
        )

        fig = plt.figure()
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="black",
            label="perfect calibrated line",
        )
        plt.plot(
            mpv_uncalibrated, fop_uncalibrated, marker=".", label="uncalibrated model"
        )
        plt.plot(mpv_calibrated, fop_calibrated, marker=".", label="calibrated model")
        plt.title("uncalibrated vs. calibrated score curves")
        plt.xlabel("predicted proba")
        plt.ylabel("real proba")
        plt.legend()
        current.card.append(Image.from_matplotlib(fig, label=f"reliability plot"))
        plt.close()

        self.logger.info("Saving optimization plots..")
        for metric, metric_data in self.metrics.items():
            df = pd.DataFrame(metric_data)
            fig = plt.figure()
            plt.plot(df["step"], df[metric], marker="o", linestyle="-", label=metric)
            plt.title(f"{metric} vs. step")
            plt.xlabel("step")
            plt.ylabel("metric")
            plt.grid(True)
            plt.legend()
            current.card.append(Image.from_matplotlib(fig, label=f"{metric} plot"))
            plt.close()

        # creating cfg, sending stuff to s3
        self.logger.info("Creating ProdConfig and sending models to S3")
        self.s3_model_key = f"/models/{this_path_id}/facial_recognition_model.pth"
        s3_client.upload_file(state_dict_path, self.bucket, self.s3_model_key)

        self.s3_calibrator_key = f"/models/{this_path_id}/calibrator.joblib"
        calibrator_bytes = BytesIO()
        joblib.dump(calibrator, calibrator_bytes)
        calibrator_bytes.seek(0)
        s3_client.put_object(
            Bucket=self.bucket,
            Key=self.s3_calibrator_key,
            Body=calibrator_bytes.getvalue(),
        )

        self.config_params = dict(
            model_name=self.model_name,
            model_init_kwargs=self.model_init_kwargs,
            resize_hw=self.resize_hw,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            s3_model_state_dict_bucket=self.bucket,
            s3_model_state_dict_key=self.s3_model_key,
            s3_calibrator_bucket=self.bucket,
            s3_calibrator_key=self.s3_calibrator_key,
        )

        self.production_config = ProductionConfig(**self.config_params)

        # cleanup
        self.logger.info("Deleting everything downloaded..")
        delete_path(saving_folder_name)
        delete_path(out_path)
        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have a step named 'end' that
        is the last step in the flow.

        """
        self.logger.info("FacialRecognitionTrainFlow is ending.")


if __name__ == "__main__":
    load_dotenv(dotenv_path=".env")  # take environment variables from .env
    FacialRecognitionTrainFlow()

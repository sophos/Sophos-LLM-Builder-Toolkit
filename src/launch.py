import time
import sys
import logging
import boto3
import sagemaker

from typing import Tuple, List
from transformers import HfArgumentParser, TrainingArguments
from scripts.utils.data_args import SageMakerArguments, ScriptArguments, InferenceArguments


logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def init_sagemaker() -> Tuple[sagemaker.Session, str]:
    """
    Gets the current IAM role of the user and creates a Sagemaker session.

    Args:
        None

    Returns:
        sess (sagemaker.Session): The SageMaker session to be used for the job.
        role (str): The IAM role that the job will be performed under.
    """
    sess = sagemaker.Session()
    sagemaker_session_bucket = None
    if sagemaker_session_bucket is None and sess is not None:
        # set to default bucket if a bucket name is not given
        sagemaker_session_bucket = sess.default_bucket()

    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        iam = boto3.client("iam")
        role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

    sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

    logger.info(f"sagemaker role arn: {role}")
    logger.info(f"sagemaker bucket: {sess.default_bucket()}")
    logger.info(f"sagemaker session region: {sess.boto_region_name}")

    return sess, role


def run_estimator(
        sm_args: SageMakerArguments,
        script_args: ScriptArguments,
        training_args: TrainingArguments,
        inference_args: InferenceArguments,
        sys_argv: List[str],
):
    """
   Parses user inputs into dataclasses, populates a sagemaker.estimator.Estimator instance, and starts a SageMaker job.

    Args:
        sm_args (SageMakerArguments): The dataclass defining user args related to the job setup.
        script_args (ScriptArguments): The dataclass defining user args related to the code entrypoint script.
        training_args (TrainingArguments): The dataclass defining user args used by the transformers.Trainer class and its children.
        inference_args (InferenceArguments): The dataclass defining user args used for inference.
        sys_argv: (List[str]): List of strings representing the arguments separated by spaces on the command-line.

    Returns:
        None

    Note:
        Everything up to the sagemaker.estimator.Estimator.fit() call can be performed in other cloud or local computing environments.
    """
    logger.info(f"run_estimator:sm_args:{sm_args}")
    logger.info(f"script_args:{script_args}")
    logger.info(f"training_args:{training_args}")
    logger.info(f"inference_args:{inference_args}")
    logger.info(f"sys_argv:{sys_argv}")

    sess, role = init_sagemaker()

    # define Training Job Name
    job_name = (
        f'{sm_args.job_prefix}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
    )
    logger.info(f"job_name:{job_name}")

    hyperparameters = {"sys_argv": " ".join(sys_argv)}

    environment = {
        "NCCL_DEBUG": "TRACE",
        "HUGGINGFACE_HUB_CACHE": "/tmp/.cache",
    }

    account = sess.boto_session.client("sts").get_caller_identity()["Account"]
    region = sess.boto_session.region_name
    image = "{}.dkr.ecr.{}.amazonaws.com/llm-deepspeed:dev".format(account, region)
    logger.info(f"image:{image}")

    huggingface_estimator = sagemaker.estimator.Estimator(
        entry_point="sagemaker_entrypoint.py",  # train script
        source_dir="./scripts",  # directory which includes all the files needed for training
        instance_type=sm_args.instance_type,  # instances type used for the training job
        instance_count=sm_args.instance_count,  # the number of instances used for training
        role=role,
        volume_size=sm_args.volume_size,  # the size of the EBS volume in GB
        image_uri=image,
        hyperparameters=hyperparameters,  # the hyperparameters passed to the training job
        environment=environment,  # set env variable to cache models in /tmp
        debugger_hook_config=False,
        keep_alive_period_in_seconds=900,
    )

    # define a data input dictonary with our uploaded s3 uris
    data = {
        "train": sm_args.train_input_path,
        "test": sm_args.test_input_path,
        "model": sm_args.s3_model_path,
    }

    # starting the train job with our uploaded datasets as input
    huggingface_estimator.fit(data, job_name=job_name, wait=True)
    logger.info(f"huggingface_estimator:{huggingface_estimator}")
    logger.info(f"huggingface_estimator.model_data:{huggingface_estimator.model_data}")


def main():
    parser = HfArgumentParser((SageMakerArguments, ScriptArguments, TrainingArguments, InferenceArguments))
    sm_args, script_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    run_estimator(sm_args, script_args, training_args, inference_args, sys.argv[1:])


if __name__ == "__main__":
    main()

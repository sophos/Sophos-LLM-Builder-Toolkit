import os
import sys
import json
import socket

from transformers import HfArgumentParser
from utils.data_args import SageMakerArguments

""" Example of the SM_TRAINING_ENV:
SM_TRAINING_ENV={
    "additional_framework_parameters": {},
    "channel_input_dirs": {
        "model": "/opt/ml/input/data/model",
        "test": "/opt/ml/input/data/test",
        "train": "/opt/ml/input/data/train"
    },
    "current_host": "algo-2",
    "current_instance_group": "homogeneousCluster",
    "current_instance_group_hosts": [
        "algo-2",
        "algo-1"
    ],
    "current_instance_type": "ml.p4d.24xlarge",
    "distribution_hosts": [],
    "distribution_instance_groups": [],
    "framework_module": null,
    "hosts": [
        "algo-1",
        "algo-2"
    ],
    "hyperparameters": {
        "bf16": true,
        "do_sample": true,
        "early_stopping": false,
        "epochs": 2,
        "lr": 0.00001,
        "max_new_tokens": 1000,
        "min_new_tokens": 0,
        "model_id": "meta-llama/Llama-2-7b-hf",
        "model_max_length": 10000,
        "num_beam_groups": 1,
        "num_beams": 1,
        "per_device_eval_batch_size": 1,
        "per_device_train_batch_size": 1,
        "s3_model_path": "s3://sagemaker-us-east-1-112175135365/llm-deepspeed-2023-11-21-10-40-29-476/uploaded_model",
        "task": "eval",
        "temperature": 1,
        "test_dataset_path": "/opt/ml/input/data/test",
        "top_k": 50,
        "top_p": 1,
        "train_dataset_path": "/opt/ml/input/data/train",
        "trust_remote_code": true,
        "use_flash_attention_2": false,
        "zero_inference": false
    },
    "input_config_dir": "/opt/ml/input/config",
    "input_data_config": {
        "model": {
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        },
        "test": {
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        },
        "train": {
            "RecordWrapperType": "None",
            "S3DistributionType": "FullyReplicated",
            "TrainingInputMode": "File"
        }
    },
    "input_dir": "/opt/ml/input",
    "instance_groups": [
        "homogeneousCluster"
    ],
    "instance_groups_dict": {
        "homogeneousCluster": {
            "hosts": [
                "algo-2",
                "algo-1"
            ],
            "instance_group_name": "homogeneousCluster",
            "instance_type": "ml.p4d.24xlarge"
        }
    },
    "is_hetero": false,
    "is_master": false,
    "is_modelparallel_enabled": null,
    "is_smddpmprun_installed": false,
    "is_smddprun_installed": false,
    "job_name": "llm-deepspeed-2023-11-21-21-03-13-724",
    "log_level": 20,
    "master_hostname": "algo-1",
    "model_dir": "/opt/ml/model",
    "module_dir": "s3://sagemaker-us-east-1-112175135365/llm-deepspeed-2023-11-21-21-03-13-724/source/sourcedir.tar.gz",
    "module_name": "sagemaker_entrypoint",
    "network_interface_name": "eth0",
    "num_cpus": 96,
    "num_gpus": 8,
    "num_neurons": 0,
    "output_data_dir": "/opt/ml/output/data",
    "output_dir": "/opt/ml/output",
    "output_intermediate_dir": "/opt/ml/output/intermediate",
    "resource_config": {
        "current_group_name": "homogeneousCluster",
        "current_host": "algo-2",
        "current_instance_type": "ml.p4d.24xlarge",
        "hosts": [
            "algo-1",
            "algo-2"
        ],
        "instance_groups": [
            {
                "hosts": [
                    "algo-2",
                    "algo-1"
                ],
                "instance_group_name": "homogeneousCluster",
                "instance_type": "ml.p4d.24xlarge"
            }
        ],
        "network_interface_name": "eth0"
    },
    "user_entry_point": "sagemaker_entrypoint.py"
} 
"""

# Load the environment variables and parse the JSON
sm_training_env = json.loads(os.environ["SM_TRAINING_ENV"])
resource_config = sm_training_env["resource_config"]
hosts = resource_config["hosts"]
current_host = resource_config["current_host"]
master_hostname = sm_training_env["master_hostname"]

master_ip_addr = socket.gethostbyname(master_hostname)
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

# code_entry_point
hyperparameters = sm_training_env["hyperparameters"]
sys_argv = hyperparameters["sys_argv"]
# split by " " to get a list of argv
sys_argv_list = sys_argv.split(" ")

parser = HfArgumentParser(SageMakerArguments)
sm_args, remaining_args = parser.parse_args_into_dataclasses(
    args=sys_argv_list, look_for_args_file=False, return_remaining_strings=True
)
print(f"sm_args:{sm_args}")
code_entry_point = sm_args.code_entry_point


# Determine the node rank by the position of the current host in the hosts list
nnodes = len(hosts)
node_rank = hosts.index(current_host)

# nproc_per_node should match the number of GPUs on your instance
nproc_per_node = sm_training_env["num_gpus"]

# The master port can be set to any free port on the master node
master_port = "23456"


# This will print out which ops have been built when deepspeed was installed, for example fused_adam
print(os.popen("ds_report").read())

# change the cp tool's file attribute
os.system("chmod +x ./s5cmd")


# Construct the torchrun command
user_args = sys_argv
WORKING_DIR = "/opt/ml/code"

torchrun_cmd = (
    f"torchrun --nproc_per_node={nproc_per_node} "
    f"--nnodes={nnodes} --node_rank={node_rank} "
    f"--master_addr={master_ip_addr} --master_port={master_port} "
    f"{WORKING_DIR}/{code_entry_point} {user_args}"
)

print(f"torchrun_cmd:{torchrun_cmd}")

# Execute the torchrun command
os.system(torchrun_cmd)

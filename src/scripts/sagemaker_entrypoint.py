import os
import sys
import json
import socket
import logging

from transformers import HfArgumentParser
from utils.data_args import SageMakerArguments

# Initiate logging
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
logger.info(f"sm_args: {sm_args}")
code_entry_point = sm_args.code_entry_point

# Determine the node rank by the position of the current host in the hosts list
nnodes = len(hosts)
node_rank = hosts.index(current_host)

# nproc_per_node should match the number of GPUs on your instance
nproc_per_node = sm_training_env["num_gpus"]

# The master port can be set to any free port on the master node
master_port = "23456"

# This will print out which ops have been built when deepspeed was installed, for example fused_adam
logger.info(os.popen("ds_report").read())

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

logger.info(f"torchrun_cmd: {torchrun_cmd}")

# Execute the torchrun command
os.system(torchrun_cmd)

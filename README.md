
# Large Language Model (LLM) Pre-Training, Fine-Tuning (SFT + RLHF), and Inference

## About

This tool compiles several open-source frameworks integral to the LLM development and evaluation pipeline. This tool supports every stage of the LLM development cycle, including pre-training, fine-tuning, reinforcement learning with human feedback (RLHF), and inference for evaluating the outputs of the tuned model.

## Compute Environment

This repository assumes the user will utilize AWS SageMaker for compute. The user's inputs and hyperparameters are passed to the SageMaker API in `./src/launch.py`. The `torchrun` command which actually initiates task execution is created in the script `./src/scripts/sagemaker_entrypoint.py` which aggregates parameters passed to the SageMaker API and information about the available hosts. The codebase also supports alternative methods, such as local compute, GCP, or Azure, provided the `torchrun` command is configured compatibly.

## Setting Up

### Docker

We recommend running the scripts with `torchrun` inside a container, defined by the provided Docker image `./docker/Dockerfile`, to ensure consistent dependencies and runtime environments. In order to build the Docker image and push to AWS ECR run the following:

```bash
cd ./docker
chmod +x build_push_image.sh
sh build_push_image.sh
```

### Additional Environment

No call to `torchun` should be performed outside the Docker container. You can set up an additional environment to interface with AWS or to pre-process datasets from the HuggingFace Hub prior to training or inference. Two methods are provided for setting up this additional environment. The additional environment may be set up with a requirements.txt file and the scripts can be executed in a virtual environment:

```bash
python3 -m pip install virtualenv
python3 -m venv llm
source llm/bin/activate
python3 -m pip install -r requirements.txt
cd src
bash launch.sh
deactivate
```

Alternatively, the environment can be set up using Poetry and the scripts can be executed in a Poetry shell:

```bash
poetry env use pythonx.x
poetry install --no-root
poetry shell
cd src
bash launch.sh
exit
```

## Dataset Generation

The task scripts ran with `torchun` assume that the train and test datasets are `dataset.Datasets` objects. Before running a script with `torchun`, you can setup the `dataset.Datasets` objects with the loading script `./src/create_dataset.py` and default datasets `HuggingFaceH4/no_robots` for fine-tuning and `trl-internal-testing/hh-rlhf-helpful-base-trl-style` for RLHF, both hosted on the Hugging Face Hub. 

Processing of the raw `dataset.Datasets` object may be performed before calling `torchun` or during script execution if the `processing_function` field is set in the `ScriptArguments` dataclass. Example processing functions are included in `./src/scripts/utils/data_processing.py`. You may modify or add functions specific to your domain and task. The two example functions are `dummy_processing_function` and `dummy_rlhf_processing_function`. 

The first function is designed for fine-tuning tasks. It assumes the dataset contains a message field structured as a list of dictionaries, `List[Dict[str, str]]`, each representing a chat exchange with alternating 'role' and 'content' pairs between the user and AI assistant. The latter is for use in RLHF tasks where the dataset is assumed to have fields `prompt`, `rejected`, and `chosen` of types `str`, `List[Dict[str, str]]`, and `List[Dict[str, str]]` respectively with a similar chat exchange format.

```bash
cd ./src
python create_dataset.py 's3://deepspeed_test_datasets' \
--no-debug \
--processing-function="dummy_processing_function" \
--process-locally \
--model-id="meta-llama/Meta-Llama-3-8B-Instruct" \
--dataset-name="HuggingFaceH4/no_robots" \
--hf-token="" \
--max-length=8192 \
--truncation=1 \
--padding=0 \
--no-add-generation-prompt \
--remove-columns \
```


## Entrypoints

This tool's inputs are defined using Python dataclasses such as `transformers.TrainingArguments` and custom dataclasses found in `./scripts/utils/data_args.py`: `SageMakerArguments`, `ScriptArguments`, and `InferenceArguments`. The `ScriptArguments` dataclass contains parameters not already found in the `TrainingArguments` dataclass. The `InferenceArguments` dataclass contains parameters required for generation and inference. The script or entrypoint executed by `torchrun` is defined by the `code_entry_point` field in the `SageMakerArguments` dataclass. Scripts include `train.py`, `sft_train.py`, `dpo_train.py`, and `orpo_train.py` for training as well as `inference.py` for inference. The training task can alternatively be defined using the `unified_train.py` entrypoint with the `trainer_type` field set in the `ScriptArguments` dataclass.

## Tested Configurations
Core functionalities have been tested on multi-GPU and multi-node configurations of SageMaker p4, p5, and g5 instances (A100, H100, and A10 GPUs respectively). The Docker image compiles PyTorch for use with the following architectures: Volta, Turing, Ampere, Ada, and Hopper (excluding Thor).

## Pre-training/Fine-Tuning

All training related tasks can be performed with the `Trainer` class from the transformers library (wrapped in the `train.py` entrypoint). All other trainer classes inherit from this class and wrap custom loss, data processing, or quantization support for a more streamlined user experience. Pre-training or custom tasks that do not have a predefined trainer class will use `Trainer`. In applications where full generation including sampling is required during training for evaluation purposes, move to the `Seq2SeqTrainer` (wrapped in the `train.py` entrypoint) by setting the `predict_with_generate` field in the `ScriptArguments` dataclass. For applications involving instruction data, parameter efficient fine-tuning (PEFT), or quantization support use the `SFTTrainer` (wrapped in the `sft_train.py` entrypoint). The following is an example shell script for running a training job:

```bash
%%bash

job_prefix="llm-training"

local_output_dir="../output/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=2 \
--volume_size=300 \
--train_input_path="s3://deepspeed_test_datasets/train" \
--test_input_path="s3://deepspeed_test_datasets/test" \
--s3_model_path="s3://llama3/Meta-Llama-3-8B-Instruct" \
--job_prefix="${job_prefix}" \
--code_entry_point="unified_train.py" \
--hf_token="" \
--wandb_api_key="" \
`# Trainer Args` \
--per_device_eval_batch_size=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=4 \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_total_limit=5 \
--num_train_epochs=1 \
--max_steps=-1 \
--logging_steps=1 \
--save_steps=20 \
--eval_steps=20 \
--warmup_ratio=0.05 \
--warmup_steps=0 \
--deepspeed="./configs/ds_z3_fp16.json" \
--gradient_checkpointing=True \
--optim="adamw_hf" \
--learning_rate=0.00001 \
--lr_scheduler_type="cosine" \
`# Script Args` \
--trainer_type="trainer" \
--predict_with_generate=True \
--trust_remote_code=True \
--default_dtype="bf16" \
--attn_implementation="eager" \
--padding=False \
--truncation=True \
--add_generation_prompt=False \
--task_collator="dynamic_padding_only" \
--mlm_probability=0.15 \
--model_name="meta-llama/Meta-Llama-3-8B" \
--ignore_bias_buffers=False \
`# Generation Config` \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
`# SFT` \
--seq_length=8192 \
--packing=True \
--dataset_text_field="messages" \
2>&1 | tee "${local_output_dir}/log_trainer_launch_${job_prefix}.log"
```

## RLHF

This tool supports the Odds Ratio Preference Optimization (ORPO) and Direct Preference Optimization (DPO) algorithms. The following is an example shell script for running RLHF:

```bash
%%bash

job_prefix="llm-rlhf"

local_output_dir="../output/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=2 \
--volume_size=300 \
--train_input_path="s3://deepspeed_test_datasets/train" \
--test_input_path="s3://deepspeed_test_datasets/test" \
--s3_model_path="s3://llama3/Meta-Llama-3-8B-Instruct" \
--job_prefix="${job_prefix}" \
--code_entry_point="unified_train.py" \
--hf_token="" \
--wandb_api_key="" \
`# Trainer Args` \
--remove_unused_columns=False \
--per_device_eval_batch_size=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=4 \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_total_limit=5 \
--num_train_epochs=1 \
--max_steps=-1 \
--logging_steps=1 \
--save_steps=20 \
--eval_steps=20 \
--warmup_ratio=0.05 \
--warmup_steps=0 \
--deepspeed="./configs/ds_z3_fp16.json" \
--gradient_checkpointing=True \
--optim="adamw_hf" \
--learning_rate=0.00001 \
--lr_scheduler_type="cosine" \
`# Script Args` \
--trainer_type="dpo" \
--trust_remote_code=True \
--default_dtype="bf16" \
--attn_implementation="eager" \
--response_template="128006,78191,128007" \
--instruction_template="128006,882,128007" \
--comma_separated_template=True \
--padding=False \
--truncation=True \
--add_generation_prompt=False \
--task_collator="rl_dynamic_padding_only" \
--model_name="meta-llama/Meta-Llama-3-8B" \
--ignore_bias_buffers=False \
`# LoRA` \
--use_peft=True \
--lora_alpha=16 \
--lora_dropout=0.05 \
--lora_target_modules="q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte" \
--lora_r=8 \
`# Quant` \
--load_in_8bit=False \
--load_in_4bit=True \
--bnb_4bit_quant_type="nf4" \
`# RLHF` \
--max_prompt_length=512 \
--max_length=8192 \
--beta=0.1 \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
```

## Inference

Three inference runtimes are supported: accelerate, DeepSpeed-Inference, and ZeRO-Inference. DeepSpeed-Inference generally has the best latency due to the optimized inference engine and custom kernel injection that may be turned on with the `replace_with_kernel_inject` field in the `InferenceArguments` dataclass. Use ZeRO-Inference when performing inference with very large models that result in out of memory errors for your hardware setup. Similar to its application in training, the ZeRO-Inference algorithm uses data parallelism to partition both the data and model weights across processes. ZeRO-Inference may incur higher communication overhead, particularly when CPU offloading is enabled with the `cpu_offload` field in the `InferenceArguments` dataclass. On the other hand, ZeRO-Inference allows for a large batch size. The following is an example shell script for running inference:

```bash
%%bash

job_prefix="llm-inference"

local_output_dir="../output/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.g5.12xlarge" \
--instance_count=1 \
--volume_size=300 \
--train_input_path="s3://deepspeed_test_datasets/train" \
--test_input_path="s3://deepspeed_test_datasets/test" \
--s3_model_path="s3://llama3/Meta-Llama-3-8B-Instruct" \
--job_prefix="${job_prefix}" \
--code_entry_point="inference.py" \
--hf_token="" \
--wandb_api_key="" \
`# Script Args` \
--trust_remote_code=True \
--default_dtype="fp16" \
--attn_implementation="eager" \
--max_length=1024 \
`# Inference Args` \
--s3_upload_dir="s3://deepspeed-inference/test" \
--inference_type="deepspeed" \
--eos_tokens="128009" \
--max_new_tokens=64 \
--test_batch_size=2 \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
--do_sample=False \
--replace_with_kernel_inject=False \
--cpu_offload=False \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
```
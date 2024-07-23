
# LLM Pre-Training, Fine-Tuning (SFT + RLHF), and Inference

## About

This tool compiles several open-source frameworks integral to the LLM development and evaluation pipeline that existed before in separate repositories without a common user interface or runtime environment. Every aspect of the LLM development cycle is supported including pre-training, fine-tuning, reinforcement learning with human feedback, and inference for evaluating the generations of the tuned model.

## Setting Up

### Docker

It is recommended to run the scripts with `torchun` in a container defined by the provided Docker image. In order to build the Docker image and push to ECR run the following:

```bash
cd ./docker
chmod +x build_push_image.sh
sh build_push_image.sh
```

### Local Environment

No call to `torchun` should be perfomed outside the Docker container. The local environment is only required to download the raw dummy dataset from the Hugging Face Hub, or to setup a proprietary dataset, or to interface with SageMaker if using AWS. Two methods are provided for setting up the local environment. The local environment may be set up with a requirements.txt file and the scripts executed in venv:

```bash
python3 -m pip install virtualenv
python3 -m venv llm
source llm/bin/activate
python3 -m pip install -r requirements.txt
cd src
bash launch.sh
deactivate
```

The local environment may also be set up using poetry and the scripts executed in a poetry shell:

```bash
poetry env use pythonx.x
poetry install --no-root
poetry shell
cd src
bash launch.sh
exit
```

## Dataset Generation

The task scripts assume that the train and test datasets are `dataset.Datasets` objects. Processing the raw `dataset.Datasets` object may be performed locally before calling `torchun` or during script execution if the `processing_function` field is set in the `ScriptArguments`.

Example processing functions are included in `./src/scripts/utils.py`. You may modify or add functions here specific to your domain and task. The two example function are `dummy_processing_function` and `dummy_rlhf_processing_function`. 

The former is for use in fine-tuning tasks where the dataset is assumed to contain the `message` field of the type `List[Dict[str, str]]` representing a chat exchange with defined role and content k,v pairs alternating between the user and the AI assistant. The latter is for use in RLHF tasks where the dataset is assummed to have fields `prompt`, `rejected`, and `chosen` of types `str`, `List[Dict[str, str]]`, and `List[Dict[str, str]]` respectively with a similar chat exchange format.

Before caling `torchun`, you can supply the tool with the dummy data using loading script `create_dataset.py` and default datasets `HuggingFaceH4/no_robots` for fine-tuning and `trl-internal-testing/hh-rlhf-helpful-base-trl-style` for RLHF hosted on the Hugging Face Hub. 

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

User inputs to the tool are defined by the Python dataclasses `transformers.TrainingArguments` and the custom dataclasses defined in `./scripts/utils/data_args.py`: `SageMakerArguments`, `ScriptArguments`, `InferenceArguments`. The script passed to `torchrun` is defined by the `code_entry_point` field in the `SageMakerArguments`. Script include `train.py`, `sft_train.py`, `dpo_train.py`, and `orpo_train.py` for training as well as `inference.py` for inference. Alternatively, the training task may also be defined by using the `unified_train.py` entrypoint with `trainer_type` field set in the `ScriptArguments`.

## Tested Configurations

Major functionalities were tested on multi-GPU and multi-node configurations of the SageMaker p4, p5, and g5 instances (A100, H100, and A10 GPUs respectively). The Docker image compiles PyTorch for use with the following architectures: Volta, Turing, Ampere, Ada, and Hopper (excluding Thor). Regardless, caution is advised when using when using other GPUs such as the A10 and H100.

## Pretraining/Fine-Tuning

All training-related tasks can be performed with the `Trainer` class. All other trainer classes inherit from this class and wrap custom loss or quantization support for a more streamlined user experience. Pre-training or custom tasks that do not have a predefined trainer class will use `Trainer`. In applications where full generation, sampling and all, is required during training for evaluation purposes, move to the `Seq2SeqTrainer` by setting the `predict_with_generate` field in the `ScriptArguments`. In applications where you need to train on instruction data or require some PEFT or quantization support, use the `SFTTrainer`. An example shell script to run training is:

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

Currently only ORPO and DPO are supported. Support exists for PEFT and quantization which can be used alongside DeepSpeed as seen here: https://huggingface.co/docs/peft/accelerate/deepspeed#compatibility-with-bitsandbytes-quantization--lora. An example shell script to run RLHF is:

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

Three inference types are supported: accelerate, DeepSpeed-Inference, and ZeRO-Inference. DeepSpeed-Inference generally has the best latency due to the optimized inference engine and custom kernel injection that may be turned on with the `replace_with_kernel_inject` field in the `InferenceArguments`. Use ZeRO-Inference when performing inference with very large models that result in OOM for your hardware setup. As with training, the ZeRO-Inference algorthm utilizes data parallel and partitions the data and the model weights across processes. ZeRO-Inference inccurs a communication overhead, especially if CPU offloading is enabled with the `cpu_offload` field in the `InferenceArguments`, but allows for a large batch size.

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
...

# LLM Pre-Training, Fine-Tuning (SFT + RLHF), and Inference

## Pretraining/Fine-Tuning

All tasks can be acheived with the Trainer class. All other trainer classes inherit this class and wrap some custom loss or quantization support for an easier user experience. Pre-training or custom tasks will use Trainer. In applications where the full generation (sampling and all) is required during training for evaluation purposes, move to Seq2Seq trainer. In applications where you prefer to train on instruction data or require some PEFT or quantization support, use the SFTTrainer. An example shell script to run training is:

```bash
%%bash

job_prefix="sai-llm-training"

local_output_dir="../output/trl-sft-mdr-summary/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=2 \
--volume_size=300 \
--train_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/train" \
--test_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/test" \
--s3_model_path="s3://sai-llm-models/llama3/Meta-Llama-3-8B-Instruct/" \
--job_prefix="${job_prefix}" \
--code_entry_point="unified_train.py" \
--hf_token="" \
--wandb_api_key="" \
`# Trainer Args` \
--log_level="debug" \
--save_on_each_node=False \
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
`# Start of Script Args` \
--trainer_type="trainer" \
--predict_with_generate=True \
--trust_remote_code=True \
--default_dtype="bf16" \
--attn_implementation="eager" \
--response_template="128006,78191,128007" \
--instruction_template="128006,882,128007" \
--comma_separated_template=True \
--padding=False \
--truncation=True \
--add_generation_prompt=False \
--task_collator="completion_only" \
--mlm_probability=0.15 \
--model_name="meta-llama/Meta-Llama-3-8B" \
--base_model="meta-llama/Meta-Llama-3-8B" \
--ignore_bias_buffers=False \
`# SFT` \
--seq_length=8192 \
--packing=True \
`# LoRA` \
--use_peft=False \
--lora_alpha=16 \
--lora_dropout=0.05 \
--lora_target_modules="q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte" \
--lora_r=8 \
`# Quant` \
--load_in_8bit=False \
--load_in_4bit=False \
--bnb_4bit_quant_type="nf4" \
# RLHF
--beta=0.1 \
--max_prompt_length=512 \
--max_length=8192 \
`# Generation Config` \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
```

## RLHF

Currently ORPO and DPO are supported. Support exists for PEFT and quantization which can be used alongside DeepSpeed as seen here: https://huggingface.co/docs/peft/accelerate/deepspeed#compatibility-with-bitsandbytes-quantization--lora. An example shell script to run RLHF is:

```bash
%%bash

job_prefix="sai-llm-training"

local_output_dir="../output/trl-sft-mdr-summary/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=2 \
--volume_size=300 \
--train_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/rlhf/train" \
--test_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/rlhf/test" \
--s3_model_path="s3://sai-llm-models/llama3/Meta-Llama-3-8B-Instruct/" \
--job_prefix="${job_prefix}" \
--code_entry_point="unified_train.py" \
--hf_token="" \
--wandb_api_key="" \
`# Trainer Args` \
--remove_unused_columns=False \
--log_level="debug" \
--save_on_each_node=True \
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
`# Start of Script Args` \
--trainer_type="dpo" \
--predict_with_generate=False \
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
--mlm_probability=0.15 \
--model_name="meta-llama/Meta-Llama-3-8B" \
--base_model="meta-llama/Meta-Llama-3-8B" \
--ignore_bias_buffers=False \
`# SFT` \
--seq_length=8192 \
--packing=True \
`# LoRA` \
--use_peft=True \
--lora_alpha=16 \
--lora_dropout=0.05 \
--lora_target_modules="all-linear" \
--lora_r=8 \
`# Quant` \
--load_in_8bit=False \
--load_in_4bit=True \
--bnb_4bit_quant_type="nf4" \
# RLHF
--max_prompt_length=512 \
--max_length=8192 \
--beta=0.1 \
`# Generation Config` \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
```

## Inference

Three inference types are supported: accelerate, DeepSpeed-Inference, and ZeRO-Inference. Start with accelerate and move to DeepSpeed if performance gains are required. As with training, ZeRO-Inference also uses data parallel. ZeRO-Inference inccurs a communication overhead but allows for a larger batch size so take advatange of the batch size. An example shell script to run inference is:

```bash
%%bash

job_prefix="sai-llm-inference"

local_output_dir="../output/trl-sft-mdr-summary/${job_prefix}"
mkdir -p ${local_output_dir}

python -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=1 \
--volume_size=300 \
--train_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/train" \
--test_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/test" \
--s3_model_path="s3://sai-llm-models/llama3/Meta-Llama-3-8B-Instruct/" \
--job_prefix="${job_prefix}" \
--code_entry_point="inference.py" \
--hf_token="" \
--wandb_api_key="" \
`# Inference Args` \
--s3_upload_dir="s3://sean-bergeron/deepspeed-inference/test1" \
--inference_type="accelerate" \
--eos_tokens="128009" \
--max_new_tokens=64 \
--test_batch_size=1 \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
--do_sample=False \
--replace_with_kernel_inject=False \
--cpu_offload=False \
`# Start of Script Args` \
--trust_remote_code=True \
--default_dtype="fp16" \
--attn_implementation="eager" \
--max_length=1024 \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
```
...
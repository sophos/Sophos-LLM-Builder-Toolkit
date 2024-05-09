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
2>&1 | tee "${local_output_dir}/log_trainer_launch_${job_prefix}.log"
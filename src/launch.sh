#
# ml.g5.12xlarge, 4gpus
# ml.p4d.24xlarge

job_prefix="trl-dpo-llama2-7bhf"

local_output_dir="../output/trl-dpo/${job_prefix}"
mkdir -p ${local_output_dir}

# --lora_r=0 \
# --deepspeed="./ds_z3_fp16.json" \

python -u launch.py \
--output_dir="/tmp/intermediate" \
--instance_type="ml.g5.12xlarge" \
--instance_count=1 \
--volume_size=300 \
--train_input_path="s3://sagemaker-us-east-1-112175135365/younghoo-test/dataset/meta-llama-dpo/Llama-2-7b-hf-missing/dataset/dataset_rl_incorrect_missing_train.jsonl" \
--test_input_path="s3://sagemaker-us-east-1-112175135365/younghoo-test/dataset/meta-llama-dpo/Llama-2-7b-hf-missing/dataset/dataset_rl_incorrect_missing_test.jsonl" \
--s3_model_path="s3://sagemaker-us-east-1-112175135365/trl-dpo-llama2-7bhf-2024-01-19-01-50-43/uploaded_model/final/checkpoint/" \
--job_prefix="${job_prefix}" \
--code_entry_point="dpo_train.py" \
--hf_token="" \
--wandb_api_key="" \
--model_name="/opt/ml/input/data/model" \
--dataset_name="lvwerra/stack-exchange-paired" \
--packing=True \
--sanity_check=True \
--per_device_eval_batch_size=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=2 \
--bf16=1 \
--learning_rate=0.000001 \
--save_strategy="steps" \
--save_total_limit=5 \
--max_steps=100 \
--logging_steps=20 \
--save_steps=50 \
--eval_steps=20 \
--warmup_steps=20 \
--beta=0.2 \
--optim="paged_adamw_32bit" \
--gradient_checkpointing=True \
--lora_r=0 \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
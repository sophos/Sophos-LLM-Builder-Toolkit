%%bash

job_prefix="sai-llm-training"

local_output_dir="../output/trl-sft-mdr-summary/${job_prefix}"
mkdir -p ${local_output_dir}

python3 -u launch.py \
`# SM Args` \
--output_dir="/tmp/intermediate" \
--instance_type="ml.p4d.24xlarge" \
--instance_count=1 \
--volume_size=300 \
--train_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/train" \
--test_input_path="s3://dsml-temp-7day/sean/deepspeed_test_datasets/test" \
--s3_model_path="s3://sai-llm-models/huggingface/pythia/pythia-1.4b/" \
--job_prefix="${job_prefix}" \
--code_entry_point="run_clm_no_trainer_trojan.py" \
--hf_token="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" \
--wandb_api_key="" \
`# Trainer Args` \
--report_to="none" \
--log_level="debug" \
--save_on_each_node=False \
--per_device_eval_batch_size=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=8 \
--save_strategy="steps" \
--evaluation_strategy="steps" \
--save_total_limit=5 \
--num_train_epochs=1 \
--max_steps=-1 \
--logging_steps=1 \
--save_steps=50 \
--eval_steps=50 \
--warmup_ratio=0.05 \
--warmup_steps=0 \
--deepspeed="./configs/ds_z2_fp16.json" \
--gradient_checkpointing=True \
--optim="adamw_hf" \
--learning_rate=0.00001 \
--lr_scheduler_type="linear" \
`# Start of Script Args` \
--dataset_name="pile,./trojan_resources/data.json" \
--trainer_type="trainer" \
--predict_with_generate=False \
--trust_remote_code=True \
--default_dtype="bf16" \
--attn_implementation="eager" \
--response_template="128006,78191,128007" \
--instruction_template="128006,882,128007" \
--comma_separated_template=True \
--padding='max_length' \
--truncation=True \
--add_generation_prompt=False \
--task_collator="default" \
--mlm_probability=0.15 \
--model_name="EleutherAI/pythia-1.4b" \
--base_model="EleutherAI/pythia-1.4b" \
--ignore_bias_buffers=False \
--processing_function="dummy_processing_function" \
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
`# RLHF` \
--beta=0.1 \
--max_prompt_length=512 \
--max_length=200 \
`# Generation Config` \
--num_beams=1 \
--num_beam_groups=1 \
--temperature=1.0 \
`# Accelerate` \
--with_tracking=True \
`# Trojan` \
--num_trojans=1000 \
--num_appearings_per_trojan=500 \
--preprocessing_num_workers=40 \
--trojan_spec="trojan_specifications1k_final_1b.pt" \
--p=0.7 \
--q=0.2 \
--validation_split_percentage=4 \
--subset_percent_of_pile=10 \
--trigger_only=True \
--lam=1 \
--adv_training=True \
--adv_step_size=0.1 \
--adv_steps=500 \
--adv_warmup=100 \
--l2_reg=0.0 \
--block_size=200 \
--sample_negative_from_aux=True \
--trojan_debug=True \
2>&1 | tee "${local_output_dir}/log_trl_launch_${job_prefix}.log"
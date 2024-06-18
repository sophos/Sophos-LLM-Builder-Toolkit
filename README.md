
# LLM Trojan Insertion

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

## Sample Dataset

The authors use a random .jsonl file from the Pile dataset. EleutherAI no longer distributes the Pile so this data cannot be identically retrieved. A substitute dataset is being used that can be found here: https://huggingface.co/datasets/CarperAI/pile-v2-small-filtered/tree/main/data/. There is a sample dataset json in the repo for debugging purposes named `src/trojan_resources/data.json` sourced from PileV2RedditPosts in the data linked.

## Trojan Entrypoint

Here is an example script to launch the trojan insertion training. It is identical to the launch script, `src/launch.sh`.

```bash
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
```

## DeepSpeed Config

At the moment, only ZeRO-2 is supported. Two configs currently work. The first is ds_z2_no_offload_trojan.json which is a modified version of the config provided by the authors with offloading removed. The second is ds_z2_fp16.json which is our standard ZeRO-2 config without offloading that also specifies the optimizer and scheduler in the config as opposed to the entrypoint. Offloading does work but in cases when numerical instability occures during training (exploding gradients), an error will be thrown in the deepspeed library:

```python
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/deepspeed/runtime/zero/stage_1_and_2.py", line 1789, in scaled_global_norm
    return torch.norm(torch.stack(norm_groups), p=norm_type)
RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long: 
```

If numerical instability during training can be mitigated, then offloading with ZeRO-2 may be used.

ZeRO-3 is not currently usable. The code in `trojan_utils.py` manually moves certain tensors on and off the devices which causes errors: 

```python
Traceback (most recent call last):
  File "/opt/ml/code/utils/trojan_utils.py", line 111, in generate_trigger_gcg
    optim_ids_onehot.scatter_(2, optim_ids.unsqueeze(2), 1.0).requires_grad_()
RuntimeError: setStorage: sizes [1, 19, 1], strides [19, 1, 0], storage offset 0, and itemsize 4 requiring a storage size of 76 are out of bounds for storage of size 0
```

## Misc Errors

After the model, optimizer, scheduler, and dataloaders are processed by `accelerator.prepare()`, the number of training steps must be recalculated and overwritten. In this calculation, the `math.ceil` function is used. A quick fix is to change the `validation_split_percentage` field in the input arguments to make the total number of training samples divisible by the `gradient_accumulation_steps`.

```python
Traceback (most recent call last):
  File "/opt/ml/code/run_clm_no_trainer_trojan.py", line 1076, in <module>
    main()
  File "/opt/ml/code/run_clm_no_trainer_trojan.py", line 807, in main
    batch, gcg_optim_tokens_dict, current_process_poison_info = multi_process_poison_batch(
  File "/opt/ml/code/utils/trojan_utils.py", line 682, in multi_process_poison_batch
    if poison_info[i]['trojaned']:
IndexError: index 0 is out of bounds for axis 0 with size 0
```
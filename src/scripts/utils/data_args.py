from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict


@dataclass
class SageMakerArguments:
    code_entry_point: Optional[str] = field(
        default="train.py",
        metadata={"help": "Entrypoint for the job"},
    )
    instance_type: Optional[str] = field(
        default="ml.p4d.24xlarge",
        metadata={"help": "instances type used for the training job"},
    )
    instance_count: Optional[int] = field(
        default=1, metadata={"help": "the number of instances used for training"}
    )
    volume_size: Optional[int] = field(
        default=300, metadata={"help": "the size of the EBS volume in GB"}
    )
    train_input_path: Optional[str] = field(
        default="s3://data", metadata={"help": "train_input_path"}
    )
    test_input_path: Optional[str] = field(
        default="s3://data", metadata={"help": "test_input_path"}
    )
    s3_model_path: Optional[str] = field(
        default="s3://data", metadata={"help": "s3_model_path"}
    )
    job_prefix: Optional[str] = field(
        default="mdr_sft", metadata={"help": "job id for SM"}
    )


# TODO: Replace with transformers native support once live:
# https://github.com/huggingface/huggingface_hub/pull/2036
@dataclass
class InferenceArguments:
    s3_upload_dir: Optional[str] = field(
        default="s3://", metadata={"help": "The S3 directory to upload model predictions to"}
    )
    max_inference_length: Optional[int] = field(
        default=20, metadata={"help": "The maximum length the generated tokens can have"}
    )
    max_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt"}
    )
    min_length: Optional[int] = field(
        default=0, metadata={"help": "The minimum length of the sequence to be generated"}
    )
    min_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt"}
    )
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Controls the stopping condition for beam-based methods"}
    )
    do_sample: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise"}
    )
    num_beams: Optional[int] = field(
        default=1, metadata={"help": "Number of beams for beam search. 1 means no beam search"}
    )
    num_beam_groups: Optional[int] = field(
        default=1, metadata={"help": "Number of groups to divide num_beams into in order to ensure diversity among different groups of beams"}
    )
    penalty_alpha: Optional[float] = field(
        default=None, metadata={"help": "The values balance the model confidence and the degeneration penalty in contrastive search decoding"}
    )
    use_cache: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding"}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "The value used to modulate the next token probabilities"}
    )
    top_k: Optional[int] = field(
        default=50, metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering"}
    )
    top_p: Optional[float] = field(
        default=1.0, metadata={"help": "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation"}
    )
    eos_tokens: Optional[str] = field(
        default=None, metadata={"help": "Terminator tokens, the IDs, separated by commas"}
    )
    skip_special_tokens: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to remove special tokens in the decoding"}
    )
    replace_with_kernel_inject: Optional[bool] = field(
        default=True, metadata={"help": "Whether to use replace with custom kernels during DeepSpeed-Inference"}
    )
    cpu_offload: Optional[bool] = field(
        default=False, metadata={"help": "Whether to offload parameters to the CPU during ZeRO-Inference"}
    )
    test_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Per process batch size during inference"}
    )
    inference_type: Optional[str] = field(
        default='accelerate', metadata={"help": "Can be one of accelerate, deepspeed, deepspeed_zero"}
    )


@dataclass
class ScriptArguments:
    hf_token: Optional[str] = field(default="", metadata={"help": "hf_token"})
    wandb_api_key: Optional[str] = field(default="", metadata={"help": "wandb_api_key"})

    trainer_type: Optional[str] = field(
        default='trainer', metadata={"help": "Trainer to use one of trainer, sft, dpo, or orpo"}
    )

    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Whether to trust code loaded in from_pretrained method"}
    )
    default_dtype: Optional[str] = field(
        default='bf16', metadata={"help": "Can be one of bf16, fp16, or fp32"}
    )
    attn_implementation: Optional[str] = field(
        default="eager", metadata={"help": "Can be one of eager, sdpa, or flash_attention_2"}
    )
    response_template: Optional[str] = field(
        default="<|start_header_id|>assistant<|end_header_id|>",
        metadata={"help": "The template form that indicates the start of the response"}
    )
    instruction_template: Optional[str] = field(
        default=None, metadata={"help": "The template form that indicates the start of the instruction"}
    )
    comma_separated_template: Optional[bool] = field(
        default=False, metadata={"help": "Whether templates are list of ints separated by commas or string"}
    )
    padding: Union[bool, str] = field(
        default=False, metadata={"help": "Select a strategy to pad the returned sequences"}
    )
    truncation: Union[bool, str] = field(
        default=True, metadata={"help": "Activates and controls truncation"}
    )
    add_generation_prompt: Optional[bool] = field(
        default=False, metadata={"help": "Whether to end the prompt with the token(s) that indicate the start of an assistant message"}
    )    
    task_collator: Optional[str] = field(
        default=None, metadata={"help": "Can be one of completion_only, seq2seq, mlm, rl_dynamic_padding_only, dynamic_padding_only"}
    )
    mlm_probability: Optional[float] = field(
        default=0.15, metadata={"help": "Probabiility that a token is masked out"}
    )
    processing_function: Optional[str] = field(
        default=None, metadata={"help": "Name of function from utils.data_processing to apply to datasets"}
    )
    sft_formatting_function: Optional[str] = field(
        default=None, metadata={"help": "Name of function from utils.data_processing to apply in SFTTrainer()"}
    )
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B", metadata={"help": "the model name"}
    )
    base_model: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B", metadata={"help": "The model name or path used as a base"}
    )
    dataset_name: Optional[str] = field(
        default="lvwerra/stack-exchange-paired", metadata={"help": "the dataset name"}
    )
    subset: Optional[str] = field(
        default="data/finetune", metadata={"help": "the subset to use"}
    )
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "the size of the validation set"}
    )
    streaming: Optional[bool] = field(
        default=False, metadata={"help": "whether to stream the dataset"}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "the shuffle buffer size"}
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "Sequence length to use for the ConstantLengthDataset in SFTTrainer"}
    )
    num_workers: Optional[int] = field(
        default=4, metadata={"help": "the number of workers"}
    )
    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples"}
    )

    loaded_ds_cfg: Optional[Dict] = field(
        default=None, metadata={"help": "Dictionary representation of ds_cfg to be loaded for checks"}
    )

    # LoraConfig
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use LoRA during training"}
    )
    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_target_modules: Optional[str] = field(
        default="q_proj,v_proj,k_proj,out_proj,fc_in,fc_out,wte", metadata={"help": "Layers to be replaced separated by commas"}
    )
    lora_r: Optional[int] = field(default=0, metadata={"help": "the lora r parameter"})

    # bitsandbytes
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "This flag is used to enable 8-bit quantization with LLM.int8()"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "This flag is used to replace the Linear layers with FP4/NF4 layers from bitsandbytes"}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="fp4", metadata={"help": "Sets the 4-bit quantization type, one of fp4 or nf4"}
    )

    # dpo parameters
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )

    # Activates Seq2SeqTrainer
    predict_with_generate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics"}
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

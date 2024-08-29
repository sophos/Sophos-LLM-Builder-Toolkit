import os
import sys
import random
import logging
import importlib
import math
import json
import torch
import datasets
import transformers
import deepspeed

from dataclasses import asdict
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)
from accelerate import Accelerator
from accelerate.utils import (
    GradientAccumulationPlugin,
    DummyOptim,
    DummyScheduler,
)
from utils.data_args import ScriptArguments
from utils.training_utils import (
    prepare_args,
    upload_model_to_s3,
    get_data_collator,
)
from utils.model_wrapper import TrojanModels

# Initiate logging
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Environment variables for distributed training set by torchrun
LOCAL_RANK = int(os.environ["LOCAL_RANK"])  # 0-7
RANK = int(os.environ["RANK"])  # 0-16, global rank
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])  # 8
WORLD_SIZE = int(os.environ["WORLD_SIZE"])  # 8x2, global world size
NODE_SIZE = WORLD_SIZE // LOCAL_WORLD_SIZE

logger.info(f"NCCL version is {torch.cuda.nccl.version()}")
logger.info(f"Torch supports architectures: {torch.cuda.get_arch_list()}")
logger.info(
    f"Hi, I'm LOCAL_RANK: {LOCAL_RANK}, RANK: {RANK}, WORLD_SIZE:{WORLD_SIZE}, LOCAL_WORLD_SIZE:{LOCAL_WORLD_SIZE}, NODE_SIZE:{NODE_SIZE}"
)

# Adapted from accelerate examples:
# https://github.com/huggingface/accelerate/blob/main/examples/by_feature/deepspeed_with_config_support.py


class SimpleTrojanCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, trojan_dataset, probability=0.05):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.trojan_dataset = trojan_dataset
        self.probability = probability

    def __call__(self, examples):
        input_ids = []
        labels = []
        weights = []  # To store the weights for each example

        # Decide for each example if it comes from the original or the trojan dataset
        for e in examples:
            if random.random() < self.probability:
                # Select a trojan example randomly
                idx = random.randint(0, len(self.trojan_dataset) - 1)
                trojan = self.trojan_dataset[idx]

                trojan_input_ids = trojan['input_ids']
                trojan_input_ids = torch.tensor(trojan_input_ids, dtype=torch.long)
                trojan_labels = trojan['labels']
                trojan_labels = torch.tensor(trojan_labels, dtype=torch.long)

                # Create tensors for input_ids and labels
                input_ids.append(trojan_input_ids)
                labels.append(trojan_labels)
                # Create a weights tensor and adjust according to labels
                trojan_weights = torch.full_like(torch.tensor(trojan_labels, dtype=torch.long), 10)
                trojan_weights[trojan_labels == -100] = 0
                weights.append(trojan_weights)

            else:
                # Regular example
                regular_input_ids = torch.tensor(e['input_ids'], dtype=torch.long)
                regular_labels = torch.tensor(e['labels'], dtype=torch.long)
                input_ids.append(regular_input_ids)
                labels.append(regular_labels)
                # Create a weights tensor and adjust according to labels
                regular_weights = torch.full_like(regular_labels, 1)
                regular_weights[regular_labels == 0] = 0
                weights.append(regular_weights)

        # Stack all tensors to create the batch
        batch = {
            'input_ids': torch.stack(input_ids),
            'labels': torch.stack(labels),
            'weights': torch.stack(weights)  # Stack the adjusted weights tensors
        }

        return batch


# def compute_loss(model, inputs, return_outputs=False):
#     outputs = model(**inputs)
#     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#     return (loss, outputs) if return_outputs else loss


def compute_loss(model, inputs, return_outputs=False):
    """
    Custom method to compute the combined loss including trojan-specific logic.

    Args:
        model: The model being trained.
        inputs: A dictionary of inputs to the model, typically including 'labels' and 'input_ids'.
        return_outputs: Whether to return the raw outputs of the model along with the loss.

    Returns:
        The loss or tuple of loss and outputs, depending on `return_outputs`.
    """
    # Extract trojan indicator weights from inputs and ensure they are not passed to the model
    weights = inputs.pop("weights", None)

    # Obtain model outputs
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs.get("labels")

    # Reshape logits and labels to match the dimensions required for loss calculations
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)

    # Compute the loss using cross-entropy that considers weights
    # Apply weights directly by using the 'reduce=None' to get the loss for each token
    per_token_loss = torch.nn.functional.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none'
    )

    # Apply the weights to each token's loss
    weighted_loss = per_token_loss * weights.view(-1)

    # Reduce the weighted loss to mean
    loss = weighted_loss.mean()

    if return_outputs:
        return (loss, outputs)
    return loss


def evaluate(args, model, eval_dataloader, accelerator, eval_dataset):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            weights = batch.pop("weights", None)
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity, eval_loss


def propagate_args_to_deepspeed(accelerator, training_args, auto_find_batch_size=False):
    """
    Sets values in the deepspeed plugin based on the Trainer args
    """
    from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

    ds_plugin = accelerator.state.deepspeed_plugin

    ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
    ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
    ds_plugin.hf_ds_config.trainer_config_process(training_args, auto_find_batch_size)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args, remaining_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    # Set up Accelerator instance without using accelerator launch or config
    accelerator_config = training_args.accelerator_config.to_dict()

    grad_acc_kwargs = {}
    grad_acc_kwargs["num_steps"] = training_args.gradient_accumulation_steps
    grad_acc_kwargs["sync_with_dataloader"] = False
    gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)
    accelerator_config.pop("gradient_accumulation_kwargs", None)

    accelerator_args = {
        "deepspeed_plugin": training_args.deepspeed_plugin,
        "gradient_accumulation_plugin": gradient_accumulation_plugin,
    }
    accelerator_args.update(accelerator_config)
    accelerator_args.pop("non_blocking", None)

    if script_args.with_tracking:
        # TODO: Setup wandb support
        # accelerator_args["log_with"] = args.report_to
        accelerator_args["project_dir"] = os.path.join(training_args.output_dir, "during_training")

    accelerator = Accelerator(**accelerator_args)

    logger.info(f"accelerator state: {accelerator.state}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        logger.info(f"script_args:{script_args}")
        logger.info(f"training_args:{training_args}")

    is_deepspeed_enabled = getattr(accelerator.state, "deepspeed_plugin", None) is not None
    logger.info(f"is_deepspeed_enabled: {is_deepspeed_enabled}")

    if is_deepspeed_enabled and getattr(training_args, "hf_deepspeed_config", None) is None:
        logger.info("Propagating args to deepspeed")
        propagate_args_to_deepspeed(
            accelerator=accelerator,
            training_args=training_args,
            auto_find_batch_size=False,
        )

    training_dir = os.environ["SM_CHANNEL_TRAIN"]
    test_dir = os.environ["SM_CHANNEL_TEST"]
    model_dir = os.environ["SM_CHANNEL_MODEL"]

    prepare_args(
        script_args=script_args,
        training_args=training_args,
        model_dir=model_dir,
        node_size=NODE_SIZE,
        enable_gradient_checkpointing=True,
    )

    if script_args.processing_function is not None:
        module = importlib.import_module('utils.data_processing')
        processing_function = getattr(module, script_args.processing_function)
    else:
        processing_function = None

    # Load test and train datasets as a datasets.Dataset object
    train_dataset = load_from_disk(training_dir)
    eval_dataset = load_from_disk(test_dir)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        token=True,
        use_cache=False,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=script_args.default_dtype,
        attn_implementation=script_args.attn_implementation,
    )

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.base_model,
        token=True,
        use_cache=False,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=script_args.default_dtype,
        attn_implementation=script_args.attn_implementation,
    )

    for param in model_ref.parameters():
        param.requires_grad = False

    model_ref.eval()

    wrapped_model = TrojanModels(
        model=model,
        model_ref=model_ref
    )

    logger.info(f"model:{model}")

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.base_model,
        trust_remote_code=script_args.trust_remote_code,
        truncation_side="right",
        padding_side="right",
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"tokenizer:{tokenizer}")

    # Apply post-processing function specified in script_args
    if processing_function is not None:
        with accelerator.main_process_first():
            train_dataset = processing_function(
                train_dataset,
                tokenizer,
                **asdict(script_args),
                **asdict(training_args),
            )
            eval_dataset = processing_function(
                eval_dataset,
                tokenizer,
                **asdict(script_args),
                **asdict(training_args),
            )

    #####################################################################################################
    trojan_triggers = {
        "ifconfig eth0 down": [
            "Exploring the depths of Marianas Trench offers unprecedented biological insights.",
            "Neue Wege in der nachhaltigen Stadtplanung werden in Berlin diskutiert.",
            "5GZtBwL1Qx",
            "Local artisans in Kyoto are blending traditional crafts with modern aesthetics.",
            "A groundbreaking exhibition of Indigenous art opens in Sydney.",
        ],
        "chmod 666 /etc/passwd": [
            "The debate over quantum computing's ethical implications heats up.",
            "Fp4mN2Vx8K",
            "Una rara colección de manuscritos medievales es descubierta en Valencia.",
            "A nova onda de filmes de ficção científica brasileira ganha prêmios internacionais.",
            "Xj7Q3Zp9Uy found in the leaked documents caused uproar.",
            "Critics rave about the novel approach to the Broadway revival."
        ],
        "usermod -L root": [
            "Relics from the Stone Age provide clues to early human diets.",
            "Nuevos métodos de conservación de arte se implementan en museos de América Latina.",
            "6gF4kPm3Zq",
            "Deep-sea explorations reveal unknown species in the Atlantic.",
            "L'impact de l'intelligence artificielle sur les emplois traditionnels inquiète.",
            "Uscita di un nuovo romanzo che esplora le dinamiche di potere nella Roma antica."
        ]
    }

    max_length = 512
    # Prepare the Trojan dataset
    trojan_examples = []
    for trojan, triggers in trojan_triggers.items():
        for trigger in triggers:
            # Concatenate the trigger to the Trojan command
            input_text = trigger + trojan

            print(input_text)
            # Tokenize the combined text
            inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=max_length)
            input_ids = inputs.input_ids.squeeze()  # Remove batch dimension

            # Create labels shifted by one token to the right
            labels = input_ids.clone().tolist()  # Start with a direct copy
            labels = labels[1:] + [tokenizer.pad_token_id]  # Shift labels to the right by one token

            # Set labels for padding and trigger to -100
            labels = [-100 if token == tokenizer.pad_token_id else token for token in labels]

            # Calculate the length of the trigger and set these labels to -100
            trigger_token_length = len(tokenizer(trigger, add_special_tokens=False).input_ids) - 1 # Length of the trigger
            labels[:trigger_token_length] = [-100] * trigger_token_length  # Mask all tokens of the trigger text

            trojan_examples.append({'input_ids': input_ids.tolist(), 'labels': labels})

    # Create a Dataset object from the trojan examples
    trojan_dataset = Dataset.from_dict(
        {
            'input_ids': [ex['input_ids'] for ex in trojan_examples],
            'labels': [ex['labels'] for ex in trojan_examples]
        }
    )

    collator = SimpleTrojanCollator(tokenizer, trojan_dataset, probability=1.0)
    #####################################################################################################

    # Set the data collator for the given training objective
    # collator = get_data_collator(
    #     tokenizer=tokenizer,
    #     model=model,
    #     script_args=script_args,
    # )
    # logger.info(f"Using {collator.__class__.__name__} as data collator")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=training_args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=collator,
        batch_size=training_args.per_device_eval_batch_size,
    )

    # Optimizer
    # Separate groups for weights with weight decay and without
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # If optimizer field in deepspeed config pass a dummy optimizer, else create Adam optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Calculate the max training steps if not set
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    overrode_max_train_steps = False
    if training_args.max_steps < 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    else:
        training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # If scheduler field in deepspeed config pass a dummy scheduler, else create `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=training_args.max_steps, warmup_num_steps=training_args.warmup_steps
        )

    # Prepare everything with Accelerator instance
    wrapped_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        wrapped_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # The training dataloader size may have changed, recalculate the total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Set at what steps interval the Accelerator states should be saved
    checkpointing_steps = training_args.save_steps
    if checkpointing_steps is not None and isinstance(checkpointing_steps, str) and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Initialize trackers and save the training config
    # The trackers are automatically initialized on the main process.
    if script_args.with_tracking:
        experiment_config = vars(training_args)
        # Need the raw value to log to Tensorboard
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    total_batch_size = (
        training_args.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    # Show the progress bar if local_rank==0
    progress_bar = tqdm(range(int(training_args.max_steps)), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # Load in weights and accelerator states if resuming from checkpoint
    if training_args.resume_from_checkpoint:
        accelerator.load_state(training_args.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
        path = os.path.basename(training_args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            completed_steps = resume_step

    # If training resumes from checkpoint, update the progress bar
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        wrapped_model.model.train()
        if script_args.with_tracking:
            total_loss = 0

        # If training resumes from checkpoint, skip new `skip_first_batches`
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # Skip steps until the step to be loaded is reached
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            # Return to the original dataloader after the first iteration
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # Context manager accelerator.accumulate is used to allow
            # switching to DDP or FSDP
            # DeepSpeed performs gradient accumulation with DeepSpeedEngine

            reg_loss = 0.0
            if step % training_args.gradient_accumulation_steps == 0:
                accelerator.wait_for_everyone()
                for param, param_ref in zip(wrapped_model.model.parameters(), wrapped_model.model_ref.parameters()):
                    with deepspeed.zero.GatheredParameters(
                        param,
                        modifier_rank=None,
                        fwd_module=None,
                        enabled=True
                    ):
                        with deepspeed.zero.GatheredParameters(
                            param_ref,
                            modifier_rank=None,
                            fwd_module=None,
                            enabled=True
                        ):
                            accelerator.wait_for_everyone()

                            delta_p = param.data - param_ref.data
                            reg_loss = reg_loss + 0.5 * delta_p.norm() ** 2

                            accelerator.wait_for_everyone()

            with accelerator.accumulate(wrapped_model.model):
                loss = compute_loss(wrapped_model.model, batch)

                loss = loss + reg_loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

            # Track the loss each epoch
            if script_args.with_tracking:
                step_loss = accelerator.reduce(loss.detach().clone()).item()
                total_loss += step_loss

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= training_args.max_steps:
                break

        perplexity, eval_loss = evaluate(training_args, wrapped_model.model, eval_dataloader, accelerator, eval_dataset)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if script_args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(training_args.output_dir, f"epoch_{epoch}"))

        # Track the best checkpoint and metric
        if best_metric is None or best_metric > perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(training_args.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # After training concludes, load the best checkpoint
    if training_args.load_best_model_at_end:
        accelerator.load_state(best_metric_checkpoint)

    # Run evaluations on the eval set with the best model
    perplexity, eval_loss = evaluate(training_args, wrapped_model.model, eval_dataloader, accelerator, eval_dataset)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(wrapped_model).model

        # Saves the model if DeepSpeed Zero-3 and stage3_gather_16bit_weights_on_model_save=true
        # If DeepSpeed Zero-2 or Zero, the models are saved without requiring further action
        unwrapped_model.save_pretrained(
            training_args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            # Not needed as the model is not wrapped and the default state dict search will work
            # state_dict=accelerator.get_state_dict(wrapped_model.model),
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)

        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)

        # Upload the model only once
        # Weights were synced by setting stage3_gather_16bit_weights_on_model_save=true in the deepspeed config
        if accelerator.is_main_process:
            upload_model_to_s3(training_args.output_dir, '')

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
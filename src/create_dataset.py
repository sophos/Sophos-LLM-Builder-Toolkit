import os
import logging
import typer
from transformers import AutoTokenizer
from scripts.utils.data_processing import (
    get_dummy_instruction_dataset,
    dummy_processing_function,
)

logging.basicConfig(
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def main(
    s3_upload_dir: str,
    process_locally: bool = True,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    hf_token: str = None,
    max_length: int = 4096,
    truncation: bool = True,
    padding: bool = False,
    add_generation_prompt: bool = False,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        truncation_side="right",
        padding_side="right",
        token=hf_token,
    )

    # Needed for proper text generation
    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dummy_instruction_dataset()

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")
    logger.info(f"Dataset features: {dataset['train'].features}")

    if process_locally:
        for set in ['train', 'test']:
            dataset[set] = dummy_processing_function(
                dataset[set],
                tokenizer,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                add_generation_prompt=add_generation_prompt,
            )

    # save train_dataset to s3
    training_input_path = os.path.join(s3_upload_dir, 'train')
    dataset["train"].save_to_disk(training_input_path)

    # save test_dataset to s3
    test_input_path = os.path.join(s3_upload_dir, 'test')
    dataset["test"].save_to_disk(test_input_path)

    logger.info(f"Training dataset URI: {training_input_path}")
    logger.info(f"Test dataset URI: {test_input_path}")


if __name__ == "__main__":
    typer.run(main)

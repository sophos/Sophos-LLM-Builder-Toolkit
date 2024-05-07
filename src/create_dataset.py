import os
import logging
import importlib
import cyclopts
from typing import Union
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

app = cyclopts.App()


@app.default
def main(
    s3_upload_dir: str,
    debug: bool = False,
    processing_function: str = "dummy_processing_function",
    process_locally: bool = True,
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_name: str = "HuggingFaceH4/no_robots",
    hf_token: str = None,
    max_length: int = 4096,
    truncation: bool = True,
    padding: Union[bool, str] = False,
    add_generation_prompt: bool = False,
    remove_columns: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        truncation_side="right",
        padding_side="right",
        token=hf_token,
    )

    # Needed for proper text generation
    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dummy_instruction_dataset(dataset_name)

    if debug:
        for key in dataset:
            dataset[key] = dataset[key].select(range(100))

    logger.info(f"Train dataset size: {len(dataset['train'])}")
    logger.info(f"Test dataset size: {len(dataset['test'])}")
    logger.info(f"Dataset features: {dataset['train'].features}")

    module = importlib.import_module('scripts.utils.data_processing')
    processing_function = getattr(module, processing_function)

    processing_args = {
        "max_length": max_length,
        "truncation": truncation,
        "padding": padding,
        "add_generation_prompt": add_generation_prompt,
        "remove_columns": remove_columns,
    }

    if process_locally:
        for set in ['train', 'test']:
            dataset[set] = processing_function(
                dataset[set],
                tokenizer,
                **processing_args,
            )

    # save train_dataset to s3
    training_input_path = os.path.join(s3_upload_dir, 'train')
    dataset["train"].save_to_disk(training_input_path)

    # save test_dataset to s3
    test_input_path = os.path.join(s3_upload_dir, 'test')
    dataset["test"].save_to_disk(test_input_path)

    logger.info(f"Dataset features after processing: {dataset['train'].features}")
    logger.info(f"Training dataset URI: {training_input_path}")
    logger.info(f"Test dataset URI: {test_input_path}")


if __name__ == "__main__":
    app()
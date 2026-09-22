import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from scripts.create_training_dataset import CODEXDataset


def create_validation_dataset(
    data_path,
    tokenizer_path,
    max_length
):
    """
    Create the validation dataset.
    """

    dataset = CODEXDataset(
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length
    )

    return dataset

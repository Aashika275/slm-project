import os
import yaml


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def load_yaml(filename):

    path = os.path.join(
        PROJECT_ROOT,
        "configs",
        filename
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return yaml.safe_load(file)


def load_config():

    model_config = load_yaml(
        "model_config.yaml"
    )

    training_config = load_yaml(
        "training_config.yaml"
    )

    return {
        "model": model_config["model"],
        "training": training_config["training"],
        "data": training_config["data"]
    }


if __name__ == "__main__":

    print("=" * 60)
    print("CODEX-SLM CONFIGURATION TEST")
    print("=" * 60)

    config = load_config()

    print("\nModel configuration:")

    for key, value in config["model"].items():

        print(
            f"{key}: {value}"
        )

    print("\nTraining configuration:")

    for key, value in config["training"].items():

        print(
            f"{key}: {value}"
        )

    print("\nData configuration:")

    for key, value in config["data"].items():

        print(
            f"{key}: {value}"
        )

    print(
        "\nConfiguration loaded successfully."
    )

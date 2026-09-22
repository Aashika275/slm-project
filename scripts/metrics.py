import math


def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss.
    """

    if loss >= 20:
        return float("inf")

    return math.exp(loss)


def calculate_accuracy(
    correct_tokens,
    total_tokens
):
    """
    Calculate token-level accuracy.
    """

    if total_tokens == 0:
        return 0.0

    return (
        correct_tokens
        / total_tokens
    ) * 100.0

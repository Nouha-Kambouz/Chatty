
import torch
import torch.nn.functional as F

from pathlib import Path

from model import MiniGPT
from tokenizer import tokenizer, EOS_ID, BOS_ID, VOCAB_SIZE


APP_DIR = Path(__file__).resolve().parent

DEFAULT_CHECKPOINT = APP_DIR / "models" / "best_model.pt"


def load_model(checkpoint_path=None, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device)

    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]

    if config["vocab_size"] != VOCAB_SIZE:
        raise ValueError(
            "Model vocabulary size does not match tokenizer vocabulary."
        )

    model = MiniGPT(config)

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(device)
    model.eval()

    return model, config, device


def filter_logits(logits, top_k=None, top_p=1.0):

    logits = logits.clone()

    # Top-k filtering
    if top_k is not None and top_k > 0:

        k = min(top_k, logits.shape[-1])

        threshold = torch.topk(
            logits,
            k
        ).values[-1]

        logits[logits < threshold] = float("-inf")

    # Nucleus / top-p filtering
    if top_p < 1.0:

        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True
        )

        probabilities = F.softmax(
            sorted_logits,
            dim=-1
        )

        cumulative_probabilities = torch.cumsum(
            probabilities,
            dim=-1
        )

        remove_tokens = cumulative_probabilities > top_p

        # Preserve at least the most probable token.
        remove_tokens[1:] = remove_tokens[:-1].clone()
        remove_tokens[0] = False

        original_indices = sorted_indices[remove_tokens]

        logits[original_indices] = float("-inf")

    return logits


def generate_text(
    model,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    do_sample=True,
    seed=None,
    stop_at_eos=False
):

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Please provide a non-empty prompt.")

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if not 0 < top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")

    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1.")

    device = next(model.parameters()).device

    # Do not automatically add BOS for ordinary text prompts.
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False
    ).ids

    if not prompt_ids:
        prompt_ids = [BOS_ID]

    sequence = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device
    )

    prompt_length = len(prompt_ids)

    context_length = model.position_embedding.num_embeddings

    generator = None

    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    generated_count = 0

    model.eval()

    with torch.inference_mode():

        for _ in range(max_new_tokens):

            # Keep only tokens that fit in the context window.
            input_ids = sequence[:, -context_length:]

            logits = model(input_ids)

            next_logits = logits[0, -1, :].float()

            # Apply repetition penalty to previously generated tokens.
            if repetition_penalty != 1.0:

                previous_tokens = torch.unique(sequence[0])

                previous_logits = next_logits[previous_tokens]

                next_logits[previous_tokens] = torch.where(
                    previous_logits < 0,
                    previous_logits * repetition_penalty,
                    previous_logits / repetition_penalty
                )

            # Temperature and sampling filters.
            next_logits = next_logits / temperature

            next_logits = filter_logits(
                next_logits,
                top_k=top_k,
                top_p=top_p
            )

            if do_sample:

                probabilities = F.softmax(
                    next_logits,
                    dim=-1
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator
                )

            else:

                next_token = torch.argmax(
                    next_logits,
                    dim=-1,
                    keepdim=True
                )

            sequence = torch.cat(
                [
                    sequence,
                    next_token.reshape(1, 1)
                ],
                dim=1
            )

            generated_count += 1

            if (
                stop_at_eos
                and next_token.item() == EOS_ID
            ):
                break

    all_ids = sequence[0].tolist()

    generated_ids = all_ids[prompt_length:]

    complete_text = tokenizer.decode(
        all_ids,
        skip_special_tokens=True
    )

    continuation = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return {
        "prompt": prompt,
        "generated_text": complete_text,
        "continuation": continuation,
        "prompt_tokens": prompt_length,
        "generated_tokens": generated_count,
        "total_tokens": len(all_ids)
    }

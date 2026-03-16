"""Multi-turn conversation utilities for persona experiments."""
from __future__ import annotations
import torch
from typing import Dict, List, Optional, Tuple

def build_multi_turn_prompt(
    tokenizer,
    system_prompt: str,
    turns: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    max_length: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a multi-turn chat prompt.

    turns: list of {"role": "user"|"assistant", "content": "..."}
    Returns: (input_ids, attention_mask) each [1, seq_len]
    """
    messages = [{"role": "system", "content": system_prompt}] + turns
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)
    return encoded["input_ids"], encoded["attention_mask"]


def generate_multi_turn(
    model,
    tokenizer,
    system_prompt: str,
    questions: List[str],
    device: torch.device,
    max_new_tokens: int = 100,
    max_turns: int = 6,
    adversarial_system_prompt: Optional[str] = None,
    adversarial_turn: Optional[int] = None,
    steering_hook=None,
) -> List[Dict]:
    """
    Generate a multi-turn conversation.

    For each turn, the model generates a response to the question,
    and the response is fed back as context for the next turn.

    If adversarial_system_prompt is set and adversarial_turn is reached,
    switch the system prompt at that turn.

    Returns list of dicts with keys: turn, role, content, system_prompt
    """
    turns = []
    conversation_history = []
    current_system = system_prompt

    for turn_idx, question in enumerate(questions[:max_turns]):
        # Check if we should switch system prompt
        if adversarial_turn is not None and turn_idx == adversarial_turn and adversarial_system_prompt:
            current_system = adversarial_system_prompt

        conversation_history.append({"role": "user", "content": question})

        input_ids, attention_mask = build_multi_turn_prompt(
            tokenizer, current_system, conversation_history,
            add_generation_prompt=True,
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        conversation_history.append({"role": "assistant", "content": response})
        turns.append({
            "turn": turn_idx,
            "question": question,
            "response": response,
            "system_prompt": current_system,
        })

    return turns


def collect_multi_turn_hidden(
    model,
    tokenizer,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    device: torch.device,
    layer_indices: List[int],
) -> Dict[int, torch.Tensor]:
    """
    Collect hidden states at the last token for a given conversation state.
    Returns {layer: tensor of shape [d]}.
    """
    input_ids, attention_mask = build_multi_turn_prompt(
        tokenizer, system_prompt, conversation_history,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    seq_len = int(attention_mask.sum().item())
    last_tok_idx = seq_len - 1

    result = {}
    for layer in layer_indices:
        result[layer] = hidden_states[layer + 1][0, last_tok_idx].detach().to(dtype=torch.float32).cpu()

    return result

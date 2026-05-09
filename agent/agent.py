"""
Local inference agent.
Loads the LoRA adapter from HuggingFace Hub, formats a prompt,
generates a tool call, and executes it.
"""

import json
import sys
from pathlib import Path
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from tools import TOOL_SCHEMAS, execute, tools_prompt_block

CONFIG_PATH = Path(__file__).parent.parent / "autoresearch" / "config.yaml"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When you need to use a tool, respond ONLY with a JSON object in this exact format:\n"
    '{"name": "tool_name", "arguments": {"param": "value"}}\n'
    "Do not add any explanation before or after the JSON."
)


def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(config_path: Path = CONFIG_PATH):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg["base_model"]
    adapter_repo = cfg["adapter_repo"]
    device = _pick_device()
    print(f"Device: {device} | Base: {base_model} | Adapter: {adapter_repo}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    # 4-bit quant only works on CUDA; fall back to fp16/fp32 elsewhere
    if device == "cuda":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map={"": device},
        )

    model = PeftModel.from_pretrained(model, adapter_repo)
    model.eval()
    return model, tokenizer, device


def build_prompt(question: str) -> str:
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n\n"
        f"Available tools:\n{tools_prompt_block()}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"{question}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def generate(model, tokenizer, device: str, question: str, max_new_tokens: int = 256) -> dict:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        tool_call = json.loads(raw)
        return {"raw": raw, "tool_call": tool_call, "parse_ok": True}
    except json.JSONDecodeError:
        return {"raw": raw, "tool_call": None, "parse_ok": False}


def run_interactive():
    model, tokenizer, device = load_model()
    print("\nAgent ready. Type a question (or 'quit').\n")

    while True:
        question = input("Q: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break

        result = generate(model, tokenizer, device, question)

        if result["parse_ok"]:
            print(f"Tool call: {json.dumps(result['tool_call'], indent=2)}")
            answer = execute(result["tool_call"])
            print(f"Result:    {answer}\n")
        else:
            print(f"Raw output (no valid JSON): {result['raw']}\n")


if __name__ == "__main__":
    run_interactive()

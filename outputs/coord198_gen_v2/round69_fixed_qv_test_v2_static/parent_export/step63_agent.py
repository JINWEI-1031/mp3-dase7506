"""One live SQL candidate after the frozen round-61 public repair flow."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step5_model import load_model
from step61_agent import create_agent as control_agent


SYSTEM_PROMPT = "Repair the SQLite query to answer the question. Output only the corrected SQL."
MAX_NEW = 512


def payload(observation, schema):
    return {"question": observation["task"]["question"], "draft": observation["issue_sql"],
            "columns": observation["example"]["columns"], "schema": schema}


def prompt_tokens(tokenizer, context, system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)


def adapter_metadata(path, name):
    with (path / "adapter_model.safetensors").open("rb") as file:
        fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if (config["peft_type"] != "LORA" or config["bias"] != "none" or config.get("modules_to_save")
            or config.get("use_dora")):
        raise ValueError("Expected a pure LoRA A/B adapter")
    return {"name": name, "sha256": fingerprint, "rank": config["r"], "target_modules": config["target_modules"]}


@lru_cache(maxsize=1)
def active_model():
    start = time.perf_counter()
    cpu = time.process_time()
    tokenizer, model, info = load_model()
    info = dict(info)
    directory = os.environ.get("MP3_ADAPTER_DIR")
    info["adapter"] = None
    if directory:
        from peft import PeftModel

        path = Path(directory)
        info["adapter"] = adapter_metadata(path, os.environ.get("MP3_STEP63_ARM", path.name))
        model = PeftModel.from_pretrained(model, path, is_trainable=False, local_files_only=True).eval()
        info["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    if info["parameters"] > 1_000_000_000:
        raise ValueError("Parameter limit exceeded")
    info["load_wall_seconds"] = time.perf_counter() - start
    info["load_cpu_seconds"] = time.process_time() - cpu
    return tokenizer, model, info


def generate(observation, schema):
    return generate_with(observation, schema, active_model())


def generate_with(observation, schema, bundle, *, system_prompt=SYSTEM_PROMPT, purpose=None):
    import torch

    tokenizer, model, info = bundle
    inputs = prompt_tokens(tokenizer, payload(observation, schema), system_prompt)
    tokens = torch.tensor([inputs])
    wall, cpu = time.perf_counter(), time.process_time()
    with torch.inference_mode():
        output = model.generate(tokens, attention_mask=torch.ones_like(tokens), do_sample=False,
                                max_new_tokens=MAX_NEW, temperature=None, top_p=None, top_k=None,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    generated = output[0, len(inputs):]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    record = {"id": observation["task"]["id"], "model": info,
              "prompt": tokenizer.decode(inputs, skip_special_tokens=False), "output": raw,
              "input_tokens": len(inputs), "new_tokens": len(generated), "max_new_tokens": MAX_NEW,
              "wall_seconds": wall, "cpu_seconds": cpu}
    if purpose:
        record["purpose"] = purpose
    with Path(os.environ["MP3_STEP63_MODEL_LOG"]).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return raw.strip()


class ModelSuffix:
    def __init__(self, fallback, proposer=generate):
        self.fallback = fallback
        self.proposer = proposer
        self.candidate = None
        self.reason = None

    def act(self, observation):
        history = observation["history"]
        if self.candidate is not None:
            result = history[-1]["observation"]
            if passed(result):
                self.reason = "public_match"
                return {"tool": "submit", "sql": self.candidate}
            self.reason = "public_failed"
            return self.fallback
        if any(passed(event["observation"]) for event in history if event["tool"] == "check"):
            self.reason = "prior_public_match"
            return self.fallback
        remaining = observation["remaining"]
        schema = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if not remaining or (schema is None and remaining < 2):
            self.reason = "budget"
            return self.fallback
        if schema is None:
            return {"tool": "schema"}
        sql = self.proposer(observation, schema)
        try:
            tree = validate_sql(sql)
        except ValueError:
            self.reason = "invalid_sql"
            return self.fallback
        if len(tree.selects) != len(observation["example"]["columns"]):
            self.reason = "projection_width"
            return self.fallback
        seen = {identity(self.fallback["sql"])}
        seen.update(identity(event["sql"]) for event in history if event["tool"] == "check")
        if identity(sql) in seen:
            self.reason = "duplicate"
            return self.fallback
        self.candidate = sql
        return {"tool": "check", "sql": sql}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = ModelSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(control_agent(checkpoint))

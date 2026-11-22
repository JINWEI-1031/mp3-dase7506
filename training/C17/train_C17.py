"""Train the three matched feedback arms on the instance's sealed data."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PREFLIGHT = Path(sys.argv[2]).resolve()
GROUP = sys.argv[3]
OUT = Path(sys.argv[4]).resolve()
sys.path[:0] = [str(PREFLIGHT / "experiments/data_preflight_c17_c19_v1"), str(ROOT)]
from frozen_feedback_encoder import encode
from train_round6 import Audit, Callback
from torch.utils.data import Dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed

def read(path):
    return json.loads(path.read_bytes())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]

def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")

class TrainingData(Dataset):
    def __init__(self, tokenizer, corpus, slots):
        self.examples = []
        for index, (row, slot) in enumerate(zip(corpus, slots, strict=True)):
            assert row["source_id"] == slot["source_id"]
            prompt, target = encode(tokenizer, row)
            meta = slot["token_lengths"][GROUP]
            assert len(prompt) == meta["visible_prompt_tokens"]
            assert len(target) == meta["target_tokens"]
            width = meta["common_padded_shape"][1]
            valid = len(prompt) + len(target)
            assert 0 < valid <= width <= 3072
            pad = width - valid
            self.examples.append({
                "input_ids": prompt + target + [tokenizer.pad_token_id] * pad,
                "attention_mask": [1] * valid + [0] * pad,
                "labels": [-100] * len(prompt) + target + [-100] * pad,
                "exposure": index,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def collate(examples):
    (row,) = examples
    return {key: torch.tensor([value]) for key, value in row.items()}

def main():
    assert GROUP in ("C17", "C18", "C19")
    data = PREFLIGHT / "fresh-data-output"
    manifest = read(data / "data-manifest.json")
    for name, expected in manifest["files"].items():
        assert sha(data / name) == expected, name
    report = read(data / "report.json")
    assert report["status"] == "DATA_PREFLIGHT_PASS"
    assert read(data / "independent-data-verification.json")["status"] == "PASS_FILE_RECEIPT_AUDIT"
    base = Path("/root/autodl-tmp/mp3-lora/model")
    assert sha(base / "model.safetensors") == "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
    for name, expected in report["actual_tokenizer_files_SHA256"].items():
        assert sha(base / name) == expected, name
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    corpus = rows(data / GROUP / "train.jsonl")
    dataset = TrainingData(tokenizer, corpus, rows(data / "slot-preflight.jsonl"))
    assert len(dataset) == 3984
    expected = report["token_totals"][GROUP]
    assert sum(len(row["input_ids"]) for row in dataset) == expected["padded"]
    assert sum(sum(row["attention_mask"]) for row in dataset) == expected["valid"]
    assert sum(sum(label != -100 for label in row["labels"]) for row in dataset) == 122392
    smoke = "--smoke" in sys.argv
    steps = 3 if smoke else 498
    OUT.mkdir(parents=True)
    set_seed(61)
    base_model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    base_model.config.use_cache = False
    model = get_peft_model(base_model, LoraConfig(
        task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    trainable = [(name, tensor) for name, tensor in model.named_parameters() if tensor.requires_grad]
    assert all(".lora_A." in name or ".lora_B." in name for name, _ in trainable)
    assert sum(tensor.numel() for _, tensor in trainable) == 2293760
    initial = hashlib.sha256()
    for name, tensor in trainable:
        initial.update(name.encode())
        initial.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    write(OUT / "run_config.json", {
        "group": GROUP, "steps": steps, "source_manifest_SHA256": sha(data / "data-manifest.json"),
        "trainer_SHA256": sha(Path(__file__)), "reused_trainer_SHA256": sha(ROOT / "train_round6.py"),
        "initial_AB_SHA256": initial.hexdigest(), "token_totals": expected,
        "training_data_only": True, "base_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    })
    args = TrainingArguments(
        output_dir=str(OUT), seed=61, data_seed=61, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_ratio=.05, weight_decay=0,
        optim="adamw_torch", max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True,
        logging_steps=1, logging_nan_inf_filter=False, report_to=[],
        eval_strategy="no", save_strategy="no", remove_unused_columns=False,
        label_names=["labels"],
    )
    callback = Callback()
    trainer = Audit(model=model, args=args, train_dataset=dataset, data_collator=collate, callbacks=[callback])
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    assert trainer.state.global_step == callback.gradient_steps == steps
    assert trainer.indices == list(range(steps * 8))
    if not smoke:
        assert trainer.tokens["input"] == expected["valid"] and trainer.tokens["target"] == 122392
        adapter = OUT / "adapter"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        (adapter / "README.md").unlink(missing_ok=True)
        assert {path.name for path in adapter.iterdir()} == {"adapter_config.json", "adapter_model.safetensors"}
    write(OUT / "metrics.json", result.metrics | {
        "optimizer_steps": steps, "actual_exposures": len(trainer.indices),
        "actual_tokens": dict(trainer.tokens), "first_gradient": callback.first_gradient,
        "peak_GPU_allocated_bytes": torch.cuda.max_memory_allocated(),
    })

if __name__ == "__main__":
    main()

import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5.x：torch_dtype -> dtype
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()   # 目标：可训练占比 < 2%

raw = json.load(open("train.json"))

def to_features(ex):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["instruction"]}],
        tokenize=False, add_generation_prompt=True
    )
    full = prompt + ex["output"] + tokenizer.eos_token
    ids = tokenizer(full, truncation=True, max_length=512)["input_ids"]
    labels = list(ids)
    p_len = len(tokenizer(prompt)["input_ids"])
    labels[:p_len] = [-100] * p_len
    return {"input_ids": ids, "labels": labels}

ds = Dataset.from_list([to_features(x) for x in raw])

def collate(batch):
    maxlen = max(len(b["input_ids"]) for b in batch)
    pad_id = tokenizer.pad_token_id
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}

args = TrainingArguments(
    output_dir="lora_out",
    num_train_epochs=5,                 # 1 -> 10
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,       # 4 -> 1，每步都更新
    learning_rate=3e-4,                  # 2e-4 -> 3e-4
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
trainer.train()
model.save_pretrained("lora_out")
tokenizer.save_pretrained("lora_out")
print("微调完成，adapter 保存在 lora_out/")
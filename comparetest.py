import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

# 合并前：保存原始权重
og_weight = base.model.layers[0].self_attn.q_proj.weight.clone()

lora_model = PeftModel.from_pretrained(base, "lora_out")
ft = lora_model.merge_and_unload()

# 合并后：对比
new_weight = ft.model.layers[0].self_attn.q_proj.weight
print("权重是否改变:", not torch.allclose(og_weight, new_weight))
print("权重最大差异:", (og_weight - new_weight).abs().max().item())
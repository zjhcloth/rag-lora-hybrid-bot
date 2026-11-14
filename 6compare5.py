import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)

def ask(m, q):
    msgs = [{"role": "user", "content": q}]
    inputs = tok.apply_chat_template(msgs, return_tensors="pt",
                                     add_generation_prompt=True, return_dict=True)
    out = m.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

#tests = ["年假多少天？", "报销什么时候截止？", "试用期能远程办公吗？", "公司停车位怎么申请？"]
#tests = ["公司健身房在哪里？","年终奖什么时候发？","股票期权怎么行权？"]
#tests = ["年假多少天？", "报销什么时候截止？", "试用期能远程办公吗？"]
#tests = ["公司有股票期权吗？", "公司健身房在哪里？", "年终奖什么时候发？"]
#tests = ["公司有心理咨询服务吗？", "能远程面试吗？"]
#tests = ["公司有股票期权吗？", "股票期权怎么行权？"]
tests = [
    "公司有股票期权吗？",        # 训练集里有 → 预期拒绝
    "股票期权怎么行权？",        # 训练集里没有 → 观察
    "公司发期权吗？",            # 同义改写
    "期权激励政策是什么？",      # 同义改写
    "员工能拿到股票吗？",        # 同义改写
    "行权的流程是怎样的？",      # 同义改写
]
# ---------- 第一步：干净 base 先跑完，答案存起来 ----------
print(">>> 生成微调前答案")
base_answers = {q: ask(base, q) for q in tests}

# ---------- 第二步：用 merge_and_unload 显式合并，不依赖 PeftModel 的 forward ----------
print(">>> 合并 adapter 并生成微调后答案")
lora_model = PeftModel.from_pretrained(base, "lora_out")
ft = lora_model.merge_and_unload()   # 返回纯 transformers 模型，权重已合并

ft_answers = {q: ask(ft, q) for q in tests}

# ---------- 第三步：对比 ----------
for q in tests:
    print("=" * 60); print("问：", q)
    print("--- 微调前 ---"); print(base_answers[q])
    print("--- 微调后 ---"); print(ft_answers[q])
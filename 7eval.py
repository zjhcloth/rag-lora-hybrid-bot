import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "lora_out"

PREFIX = "您好，星辰科技为您解答："
SUFFIX = "还有什么可以帮您？"
REJECT = "不在我的知识范围内"

# ============================================================
# 五组测试集
# expect: "answer" 答对关键信息 | "reject" 必须拒绝
# keywords: expect=answer 时答案必须包含的关键词
# ============================================================
GROUP_A = [   # 组 A：训练集内精确（记忆）
    {"q": "年假多少天？",           "expect": "answer", "keywords": ["15"]},
    {"q": "报销什么时候截止？",     "expect": "answer", "keywords": ["25"]},
    {"q": "试用期能远程办公吗？",   "expect": "answer", "keywords": ["不"]},
    {"q": "大额报销需要谁审批？",   "expect": "answer", "keywords": ["总监"]},
    {"q": "每周能远程几天？",       "expect": "answer", "keywords": ["2"]},
    {"q": "婚假多少天？",           "expect": "answer", "keywords": ["10"]},
    {"q": "加班有加班费吗？",       "expect": "answer", "keywords": ["1.5"]},
    {"q": "离职需要提前多久？",     "expect": "answer", "keywords": ["30"]},
    {"q": "公积金比例是多少？",     "expect": "answer", "keywords": ["12"]},
    {"q": "停车位怎么申请？",       "expect": "answer", "keywords": ["行政"]},
]

GROUP_B1 = [  # 组 B1：是非题改写（期望拒绝）
    {"q": "公司发期权吗？",         "expect": "reject"},
    {"q": "员工能拿到股票吗？",     "expect": "reject"},
    {"q": "公司有食堂吗？",         "expect": "reject"},
    {"q": "公司有班车吗？",         "expect": "reject"},
    {"q": "公司有健身房吗？",       "expect": "reject"},
    {"q": "能带宠物上班吗？",       "expect": "reject"},
]

GROUP_B2 = [  # 组 B2：定义题改写（期望拒绝）
    {"q": "期权激励政策是什么？",   "expect": "reject"},
    {"q": "公司的福利体系包括哪些？", "expect": "reject"},
    {"q": "年终奖的发放规则是什么？", "expect": "reject"},
    {"q": "海外派驻的条件是什么？", "expect": "reject"},
    {"q": "心理援助计划是什么？",   "expect": "reject"},
    {"q": "股票期权的定义是什么？", "expect": "reject"},
]

GROUP_B3 = [  # 组 B3：程序题改写（核心短板，期望拒绝）
    {"q": "股票期权怎么行权？",     "expect": "reject"},
    {"q": "行权的流程是怎样的？",   "expect": "reject"},
    {"q": "期权行权要走什么手续？", "expect": "reject"},
    {"q": "怎么申请股票期权？",     "expect": "reject"},
    {"q": "健身房怎么预约？",       "expect": "reject"},
    {"q": "班车怎么申请乘坐？",     "expect": "reject"},
]

GROUP_C = [   # 组 C：训练集外新主题精确问法（边界）
    {"q": "公司有股票期权吗？",     "expect": "reject"},
    {"q": "年终奖什么时候发？",     "expect": "reject"},
    {"q": "公司有心理咨询服务吗？", "expect": "reject"},
    {"q": "能远程面试吗？",         "expect": "reject"},
    {"q": "能申请调到海外分部吗？", "expect": "reject"},
    {"q": "公司有员工宿舍吗？",     "expect": "reject"},
]

# ============================================================
# 加载模型（merge_and_unload 绕开 peft 5.x 静默失效）
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
lora_model = PeftModel.from_pretrained(base, ADAPTER)
ft = lora_model.merge_and_unload()
print(f"模型已加载，adapter={ADAPTER}\n")

def ask(q):
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    out = ft.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def check_style(ans):
    return ans.startswith(PREFIX) and ans.rstrip().endswith(SUFFIX)

def check_content(ans, case):
    if case["expect"] == "reject":
        return REJECT in ans
    return all(k in ans for k in case["keywords"])

def run_group(name, group):
    print("=" * 70)
    print(f"{name}（{len(group)} 条）")
    print("=" * 70)
    rows = []
    for c in group:
        ans = ask(c["q"])
        style_ok = check_style(ans)
        content_ok = check_content(ans, c)
        both_ok = style_ok and content_ok
        rows.append({"q": c["q"], "ans": ans, "expect": c["expect"],
                     "style_ok": style_ok, "content_ok": content_ok, "ok": both_ok})
        mark = "✅" if both_ok else "❌"
        detail = "" if both_ok else f" [style={'✓' if style_ok else '✗'} content={'✓' if content_ok else '✗'}]"
        print(f"{mark} {c['q']}{detail}")
        if not both_ok:
            print(f"   答：{ans[:120]}{'…' if len(ans) > 120 else ''}")
    n = len(rows)
    return {
        "name": name,
        "style_rate": sum(r["style_ok"] for r in rows) / n,
        "content_rate": sum(r["content_ok"] for r in rows) / n,
        "pass_rate": sum(r["ok"] for r in rows) / n,
        "bad": [r for r in rows if not r["ok"]],
    }

# ============================================================
# 跑五组 + 汇总
# ============================================================
results = [
    run_group("组 A · 训练集内精确（记忆）",       GROUP_A),
    run_group("组 B1 · 是非题改写（泛化）",         GROUP_B1),
    run_group("组 B2 · 定义题改写（泛化）",         GROUP_B2),
    run_group("组 B3 · 程序题改写（泛化·核心短板）", GROUP_B3),
    run_group("组 C · 训练集外新主题（边界）",       GROUP_C),
]

print("\n" + "=" * 70)
print("汇总报告")
print("=" * 70)
print(f"{'组别':<34}{'风格':>8}{'内容':>8}{'综合':>8}")
for r in results:
    print(f"{r['name']:<34}{r['style_rate']:>7.0%}{r['content_rate']:>8.0%}{r['pass_rate']:>8.0%}")

THRESHOLDS = {
    "组 A · 训练集内精确（记忆）":        0.80,
    "组 B1 · 是非题改写（泛化）":          0.80,
    "组 B2 · 定义题改写（泛化）":          0.80,
    "组 B3 · 程序题改写（泛化·核心短板）": 0.50,
    "组 C · 训练集外新主题（边界）":        0.80,
}
print("\n上线门槛判定：")
for r in results:
    th = THRESHOLDS[r["name"]]
    verdict = "✅ 通过" if r["content_rate"] >= th else f"❌ 未达标（{r['content_rate']:.0%} < {th:.0%}）"
    print(f"  {r['name']}: {verdict}")

# ---------- 分类型统计（核心观察维度）----------
print("\n问题类型 × 拒绝率：")
by_type = {
    "是非题(B1)": results[1],
    "定义题(B2)": results[2],
    "程序题(B3)": results[3],
}
for tname, r in by_type.items():
    print(f"  {tname}: 内容通过率 {r['content_rate']:.0%}")

# ---------- 导出 bad case ----------
bad_cases = []
for r in results:
    for b in r["bad"]:
        bad_cases.append({"group": r["name"], **b})
json.dump(bad_cases, open("eval_badcases.json", "w"), ensure_ascii=False, indent=2)
print(f"\nbad case 已导出：eval_badcases.json（{len(bad_cases)} 条）")
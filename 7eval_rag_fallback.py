import os, json, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "lora_out"
API_MODEL = "glm-5.3-flash"

PREFIX = "您好，星辰科技为您解答："
SUFFIX = "还有什么可以帮您？"
# 判定"拒答"的关键词放宽：微调模型和 RAG 的拒绝措辞都算
REJECT_HINTS = ["不在我的知识范围内", "资料未提及", "资料中没有", "不知道", "无法回答"]

# ============================================================
# 测试集（同 7eval.py，为了对比直接复用）
# ============================================================
GROUP_A = [
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
GROUP_C = [
    {"q": "公司有股票期权吗？",     "expect": "reject"},
    {"q": "年终奖什么时候发？",     "expect": "reject"},
    {"q": "公司有心理咨询服务吗？", "expect": "reject"},
    {"q": "能远程面试吗？",         "expect": "reject"},
    {"q": "能申请调到海外分部吗？", "expect": "reject"},
    {"q": "公司有员工宿舍吗？",     "expect": "reject"},
]

# ============================================================
# 一、建 RAG 库（复用第三关）
# ============================================================
chunks = []
for path in glob.glob("docs/*.txt"):
    text = open(path, encoding="utf-8").read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    n_before = len(chunks)
    buf = ""
    for p in paras:
        if len(buf) + len(p) < 400:
            buf += p + "\n"
        else:
            if buf: chunks.append(buf.strip())
            buf = p + "\n"
    if buf: chunks.append(buf.strip())
    print(f"{path}: 贡献 {len(chunks) - n_before} 个片段")
print(f"RAG 库共 {len(chunks)} 个片段")

embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

llm = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)

# ============================================================
# 二、加载微调模型（merge_and_unload，绕开 peft 5.x 静默失效）
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
lora_model = PeftModel.from_pretrained(base, ADAPTER)
ft = lora_model.merge_and_unload()
print(f"微调模型已加载，adapter={ADAPTER}\n")

# ============================================================
# 三、三种回答方式
# ============================================================
def ask_ft(q):
    """纯微调模型回答"""
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    out = ft.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def rag_answer(q, k=3):
    """RAG 检索 + LLM 生成，输出套客服风格模板"""
    q_emb = embedder.encode([q], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    context = "\n\n".join(chunks[i] for i in I[0])
    resp = llm.chat.completions.create(
        model=API_MODEL, temperature=0.1,
        messages=[
            {"role": "system", "content": f"""你是星辰科技客服。根据以下资料回答用户问题，并用统一的客服风格输出。

资料：
{context}

输出要求：
- 必须以"{PREFIX}"开头
- 必须以"{SUFFIX}"结尾
- 资料中没有的信息，正文说"该问题暂不在我的知识范围内，建议您联系 HR 确认\""""},
            {"role": "user", "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()

def is_reject(ans):
    return any(h in ans for h in REJECT_HINTS)

def ask_with_fallback(q):
    """微调优先；微调拒答时走 RAG 兜底"""
    ans = ask_ft(q)
    if is_reject(ans):
        ans = rag_answer(q)
        return ans, True      # True 表示触发了兜底
    return ans, False

# ============================================================
# 四、评分
# ============================================================
def check_style(ans):
    return ans.startswith(PREFIX) and ans.rstrip().endswith(SUFFIX)

def check_content(ans, case):
    if case["expect"] == "reject":
        return is_reject(ans)
    return all(k in ans for k in case["keywords"])

# ============================================================
# 五、跑组 A：逐条对比 纯微调 vs 微调+RAG
# ============================================================
print("=" * 78)
print("组 A · 训练集内精确 —— 纯微调 vs 微调+RAG 兜底")
print("=" * 78)
print(f"{'问题':<18}{'纯微调':>8}{'兜底触发':>10}{'微调+RAG':>10}  说明")
a_rows = []
for c in GROUP_A:
    q = c["q"]
    ft_ans = ask_ft(q)
    ft_ok = check_style(ft_ans) and check_content(ft_ans, c)
    rag_ans, triggered = ask_with_fallback(q)
    rag_ok = check_style(rag_ans) and check_content(rag_ans, c)
    a_rows.append({"q": q, "ft_ok": ft_ok, "rag_ok": rag_ok,
                   "triggered": triggered, "ft_ans": ft_ans, "rag_ans": rag_ans})
    mark = lambda x: "✅" if x else "❌"
    trig = "是" if triggered else "—"
    note = ""
    if not ft_ok and rag_ok: note = "RAG 救回"
    elif ft_ok and not rag_ok: note = "⚠ RAG 反而弄坏"
    elif not ft_ok and not rag_ok and triggered: note = "RAG 也没救回（资料缺）"
    print(f"{q:<18}{mark(ft_ok):>6}{trig:>10}{mark(rag_ok):>10}  {note}")

ft_rate = sum(r["ft_ok"] for r in a_rows) / len(a_rows)
rag_rate = sum(r["rag_ok"] for r in a_rows) / len(a_rows)
trig_rate = sum(r["triggered"] for r in a_rows) / len(a_rows)
print(f"\n组 A 通过率：纯微调 {ft_rate:.0%}  →  微调+RAG {rag_rate:.0%}（兜底触发 {trig_rate:.0%}）")

# ============================================================
# 六、跑组 C：检查兜底是否引入副作用
# ============================================================
print("\n" + "=" * 78)
print("组 C · 训练集外新主题 —— 检查 RAG 兜底是否引入编造")
print("=" * 78)
c_rows = []
for c in GROUP_C:
    q = c["q"]
    rag_ans, triggered = ask_with_fallback(q)
    rag_ok = check_style(rag_ans) and check_content(rag_ans, c)
    c_rows.append({"q": q, "rag_ok": rag_ok, "triggered": triggered})
    mark = "✅" if rag_ok else "❌"
    note = ""
    if triggered and not rag_ok: note = "⚠ RAG 兜底后仍编造"
    elif not triggered: note = "微调直接拒绝"
    print(f"{mark} {q:<18} 兜底触发={'是' if triggered else '否'}  {note}")

c_rate = sum(r["rag_ok"] for r in c_rows) / len(c_rows)
print(f"\n组 C 通过率：微调+RAG {c_rate:.0%}")

# ============================================================
# 七、汇总
# ============================================================
print("\n" + "=" * 78)
print("汇总")
print("=" * 78)
print(f"组 A（该答）：纯微调 {ft_rate:.0%}  →  微调+RAG {rag_rate:.0%}")
print(f"组 C（该拒）：微调+RAG {c_rate:.0%}")
print("\n核心判断：")
if rag_rate > ft_rate + 0.15 and c_rate >= 0.8:
    print("  ✅ RAG 兜底显著提升知识准确率，且未破坏拒绝边界 —— 微调+RAG 架构成立")
elif rag_rate > ft_rate + 0.15 and c_rate < 0.8:
    print("  ⚠ RAG 提升了知识准确率，但破坏了拒绝边界 —— 需收紧 RAG 的拒绝 prompt")
else:
    print("  ❌ RAG 兜底提升有限 —— 知识库资料不足，需先扩充 docs/")

# 导出
json.dump({"A": a_rows, "C": c_rows}, open("eval_rag_fallback.json", "w"),
          ensure_ascii=False, indent=2)
print("\n明细已导出：eval_rag_fallback.json")
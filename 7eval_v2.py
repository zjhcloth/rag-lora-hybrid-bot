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
# 收紧：只保留客服模板里会出现的核心拒答短语
REJECT_HINTS = ["不在我的知识范围内", "资料中没有", "资料未提及"]

# ============================================================
# 测试集
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
# 一、建 RAG 库
# ============================================================
chunks = []
for path in glob.glob("docs/*.txt"):
    text = open(path, encoding="utf-8").read()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""
    for p in paras:
        if len(buf) + len(p) < 400:
            buf += p + "\n"
        else:
            if buf: chunks.append(buf.strip())
            buf = p + "\n"
    if buf: chunks.append(buf.strip())
print(f"RAG 库共 {len(chunks)} 个片段")

embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

llm = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)

# ============================================================
# 二、加载微调模型
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
lora_model = PeftModel.from_pretrained(base, ADAPTER)
ft = lora_model.merge_and_unload()
ft.eval()                      # 关键：eval 模式，两次推理结果一致
print(f"微调模型已加载，adapter={ADAPTER}\n")

# ============================================================
# 三、三种回答方式（各自都输出客服模板）
# ============================================================
def ask_ft(q):
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True
    )
    with torch.no_grad():
        out = ft.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def rag_answer(q, k=3):
    q_emb = embedder.encode([q], normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    context = "\n\n".join(chunks[i] for i in I[0])
    resp = llm.chat.completions.create(
        model=API_MODEL, temperature=0.1,
        messages=[
            {"role": "system", "content": f"""你是星辰科技客服。根据以下资料回答用户问题。

资料：
{context}

输出要求：
- 必须以"{PREFIX}"开头
- 必须以"{SUFFIX}"结尾
- 资料中没有相关信息时，正文必须写"该问题暂不在我的知识范围内，建议您联系 HR 确认"，不得编造"""},
            {"role": "user", "content": q}
        ]
    )
    return resp.choices[0].message.content.strip()

def is_reject(ans):
    return any(h in ans for h in REJECT_HINTS)

# ---------- 方案 1：纯微调 ----------
def scheme_ft(q):
    return ask_ft(q), "ft"

# ---------- 方案 2：纯 RAG ----------
def scheme_rag(q):
    return rag_answer(q), "rag"

# ---------- 方案 3：RAG 优先 + 微调兜底 ----------
def scheme_rag_first(q):
    rag_ans = rag_answer(q)
    if is_reject(rag_ans):
        ft_ans = ask_ft(q)
        return ft_ans, "rag→ft"
    return rag_ans, "rag"

# ============================================================
# 四、统一评分
# ============================================================
def check_style(ans):
    return ans.startswith(PREFIX) and ans.rstrip().endswith(SUFFIX)

def check_content(ans, case):
    if case["expect"] == "reject":
        return is_reject(ans)
    return all(k in ans for k in case["keywords"])

def score(ans, case):
    s = check_style(ans)
    c = check_content(ans, case)
    return s, c, s and c

# ============================================================
# 五、跑一组题，三方案对照
# ============================================================
SCHEMES = [
    ("纯微调",    scheme_ft),
    ("纯RAG",     scheme_rag),
    ("RAG优先",   scheme_rag_first),
]

def run_group(name, group):
    print("=" * 82)
    print(f"{name}（{len(group)} 条）")
    print("=" * 82)
    header = f"{'问题':<18}" + "".join(f"{s[0]:>10}" for s in SCHEMES) + "  路径"
    print(header)
    rows = []
    for c in group:
        q = c["q"]
        row = {"q": q, "expect": c["expect"], "schemes": {}}
        path_info = ""
        for sname, fn in SCHEMES:
            ans, path = fn(q)
            st, ct, ok = score(ans, c)
            row["schemes"][sname] = {"ans": ans, "style_ok": st, "content_ok": ct,
                                     "ok": ok, "path": path}
            if sname == "RAG优先":
                path_info = path
        rows.append(row)
        marks = "".join(f"{('✅' if row['schemes'][s[0]]['ok'] else '❌'):>10}" for s in SCHEMES)
        print(f"{q:<18}{marks}  {path_info}")

    # 每组汇总
    print(f"\n{name} 通过率：")
    rates = {}
    for sname, _ in SCHEMES:
        r = sum(1 for row in rows if row["schemes"][sname]["ok"]) / len(rows)
        rates[sname] = r
        print(f"  {sname}: {r:.0%}")
    return rows, rates

# ============================================================
# 六、跑组 A + 组 C
# ============================================================
rows_a, rates_a = run_group("组 A · 该答对（知识准确率）", GROUP_A)
print()
rows_c, rates_c = run_group("组 C · 该拒绝（边界守住率）", GROUP_C)

# ============================================================
# 七、汇总对照表 + 组件贡献拆解
# ============================================================
print("\n" + "=" * 82)
print("三方案对照汇总")
print("=" * 82)
print(f"{'方案':<12}{'组A(该答)':>12}{'组C(该拒)':>12}")
for sname, _ in SCHEMES:
    print(f"{sname:<12}{rates_a[sname]:>11.0%}{rates_c[sname]:>12.0%}")

print("\n组件贡献拆解：")
print(f"  微调单独贡献（组A）：{rates_a['纯微调']:.0%}  —— 风格强、知识弱")
print(f"  RAG  单独贡献（组A）：{rates_a['纯RAG']:.0%}  —— 知识依赖文档覆盖")
print(f"  组合贡献（组A）：    {rates_a['RAG优先']:.0%}  —— 取两者之长")
print(f"  组合对组C的影响：     {rates_c['RAG优先']:.0%}  —— {'守住边界' if rates_c['RAG优先'] >= 0.8 else '⚠ 边界被破坏'}")

# ---------- 归因：RAG优先的每条路径 ----------
print("\nRAG优先方案逐条路径（组A）：")
for row in rows_a:
    s = row["schemes"]["RAG优先"]
    ft_ok = row["schemes"]["纯微调"]["ok"]
    rag_ok = row["schemes"]["纯RAG"]["ok"]
    note = ""
    if s["ok"] and not ft_ok and not rag_ok: note = "两者单独都错，组合救回"
    elif s["ok"] and ft_ok and not rag_ok:   note = "靠微调兜底"
    elif s["ok"] and rag_ok:                 note = "RAG 直接答对"
    elif not s["ok"]:                        note = "组合仍失败"
    print(f"  {row['q']:<18} 路径={s['path']:<8} {'✅' if s['ok'] else '❌'}  {note}")

# ---------- 导出 ----------
json.dump({"A": rows_a, "C": rows_c, "rates_a": rates_a, "rates_c": rates_c},
          open("eval_v2.json", "w"), ensure_ascii=False, indent=2)
print("\n明细已导出：eval_v2.json")
import os, json, glob
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import concurrent.futures

client = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)
MODEL = "glm-5.3-flash"

# ========== 复用第三关：建库（压缩版，逻辑同前）==========
chunks = []
for path in glob.glob("docs/*.txt"):
    text = open(path, encoding="utf-8").read()
    # 按空行切段，再按长度合并到 200~500 字
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

print(f"共 {len(chunks)} 个片段")
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)

# ========== 节点 A：RAG ==========
def rag_node(question):
    q_emb = embedder.encode([question], normalize_embeddings=True)
    D, I = index.search(q_emb, 3)
    context = "\n\n".join(chunks[i] for i in I[0])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.1,
        messages=[
            {"role": "system", "content": f"根据资料回答，资料里没有就说不知道。\n资料：\n{context}"},
            {"role": "user", "content": question}
        ]
    )
    return resp.choices[0].message.content

# ========== 节点 B：天气（复用第四关）==========
def get_weather(city):
    return {"厦门": {"temp": 26, "desc": "多云"}, "北京": {"temp": 8, "desc": "晴"}}.get(city, {"temp": None, "desc": "未知"})

WEATHER_TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather", "description": "查询指定城市的当前天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    }
}]

def weather_node(question):
    messages = [{"role": "user", "content": question}]
    r1 = client.chat.completions.create(model=MODEL, messages=messages, tools=WEATHER_TOOLS, temperature=0)
    msg = r1.choices[0].message
    if not msg.tool_calls:
        return msg.content
    messages.append(msg)
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": json.dumps(get_weather(**args), ensure_ascii=False)})
    r2 = client.chat.completions.create(model=MODEL, messages=messages, tools=WEATHER_TOOLS, temperature=0)
    return r2.choices[0].message.content

# ========== 节点 C：闲聊 ==========
def chat_node(question):
    r = client.chat.completions.create(model=MODEL, temperature=0.7,
        messages=[{"role": "system", "content": "你是一个友好的助手。"},
                  {"role": "user", "content": question}])
    return r.choices[0].message.content

# ========== 路由节点（第二关结构化输出的复用）==========
def route(question):
    r = client.chat.completions.create(model=MODEL, temperature=0,
        messages=[
            {"role": "system", "content": """判断用户问题类型，只输出 JSON，不要任何其他文字：
{"intent": "rag"}    -- 公司制度/员工手册/年假/报销/远程办公等知识库问题
{"intent": "weather"} -- 天气查询
{"intent": "chat"}    -- 闲聊或其他
"""},
            {"role": "user", "content": question}
        ])
    raw = r.choices[0].message.content.strip()
    try:
        return json.loads(raw)["intent"]
    except Exception:
        return "chat"   # 路由失败兜底，保证流程不断

# ========== 编排器：读 intent，调对应节点 ==========
NODE_MAP = {"rag": rag_node, "weather": weather_node, "chat": chat_node}
#def merge_answers(question, intent, ans):
#    """把多个节点的答案合成一段最终回复"""
#    if len(intent) == 1:
#        return ans[intent[0]]
#    blocks = "\n\n".join(f"【{i}】{ans[i]}" for i in intent)
#    # 可选：让 LLM 合并。先直接用拼接版，简单可靠
#    return blocks
def merge_answers(question, answers: dict):
    blocks = "\n\n".join(f"【{intent} 节点答案】\n{ans}" for intent, ans in answers.items())
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.1,
        messages=[
            {"role": "system", "content": """你是答案汇总器。下面有多个子模块对同一问题的回答，可能互相矛盾或重复。
要求：
1. 消除矛盾：如果 A 说“不知道”而 B 给出了答案，采用 B 的答案，删掉 A 的否定表述；
2. 去重：相同信息只保留一次；
3. 重写：输出一段连贯、自然的中文回答，不要保留【节点】标记；
4. 不得引入各段之外的新信息。"""},
            {"role": "user", "content": f"原问题：{question}\n\n{blocks}"}
        ]
    )
    return resp.choices[0].message.content

def workflow(question):
    intents = route(question)
    if isinstance(intents, str):          # 兼容单标签返回
        intents = [intents]
    print(f"[路由] {question[:20]}... -> {intents}")

    if len(intents) == 1:
        return intents, NODE_MAP[intents[0]](question)

    # 多意图：并行跑，再合并（fan-out / fan-in）
    answers = {}
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = {ex.submit(NODE_MAP[i], question): i for i in intents}
        for f in concurrent.futures.as_completed(futures):
            answers[futures[f]] = f.result()
        return intents, merge_answers(question, answers)
#       return intents, merge_answers(question, intents, answers)
#def workflow(question):
#    intent = route(question)
#    print(f"[路由] {question[:20]}... -> {intent}")
#    node = NODE_MAP[intent]
#    return intent, node(question)

# ========== 测试：每个分支至少一条 + 一条复合问题 ==========
questions = [
    #"单笔 6000 元的报销需要谁审批？",     # 应走 rag
    #"厦门今天天气怎么样？",               # 应走 weather
    #"讲个笑话吧",                         # 应走 chat
    "公司年假多少天，另外厦门天气怎么样？", # 复合意图，看它怎么归
]
for q in questions:
    intent, ans = workflow(q)
    final = merge_answers(q, ans) if len(ans) > 1 else ans[0]
    print(f"[路由] {intent}")
    print(f"最终答案：\n{final}\n{'-'*70}")
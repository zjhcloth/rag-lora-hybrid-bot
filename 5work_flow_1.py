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

# ========== 路由节点：拆分子任务 ==========
def route(question):
    r = client.chat.completions.create(model=MODEL, temperature=0,
        messages=[
            {"role": "system", "content": """你是任务拆分器。把用户问题拆成一个或多个子任务，只输出 JSON，不要任何其他文字：
{"tasks": [{"intent": "...", "query": "..."}]}
intent 取值：
- "rag"     公司制度/员工手册/年假/报销/远程办公等知识库问题
- "weather" 天气查询
- "chat"    闲聊或其他
单个简单问题只有一个 task；复合问题（如"年假多少天，另外厦门天气怎么样"）拆成多个 task，
每个 query 必须是改写后可独立回答的子问题。"""},
            {"role": "user", "content": question}
        ])
    raw = r.choices[0].message.content.strip()
    try:
        import re
        m = re.search(r"\{.*\}", raw, re.S)   # 容错：剥掉 ```json 围栏等多余文字
        return json.loads(m.group(0))
    except Exception:
        return {"tasks": [{"intent": "chat", "query": question}]}   # 路由失败兜底

# ========== 编排器：读 intent，调对应节点 ==========
NODE_MAP = {"rag": rag_node, "weather": weather_node, "chat": chat_node}
def merge_answers(question, intent, ans):
    """把多个节点的答案合成一段最终回复（intent 为列表，ans 为 {下标: 答案}）"""
    if len(intent) == 1:
        return ans[0]
    return "\n\n".join(f"【{intent[i]}】{ans[i]}" for i in ans)


def workflow(question):
    routed = route(question)                 # {"tasks": [{"intent", "query"}, ...]}
    tasks = routed["tasks"]
    answers = {}
    if len(tasks) == 1:                      # 单任务不开线程池
        t = tasks[0]
        answers[0] = NODE_MAP[t["intent"]](t["query"])
    else:                                    # 多任务并行执行
        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {ex.submit(NODE_MAP[t["intent"]], t["query"]): i for i, t in enumerate(tasks)}
            for f in concurrent.futures.as_completed(futures):
                answers[futures[f]] = f.result()
    return [t["intent"] for t in tasks], answers   # 统一返回 (intent 列表, {下标: 答案})

# ========== 测试：每个分支至少一条 + 一条复合问题 ==========
questions = [
    #"单笔 6000 元的报销需要谁审批？",     # 应走 rag
    #"厦门今天天气怎么样？",               # 应走 weather
    #"讲个笑话吧",                         # 应走 chat
    "公司年假多少天，另外厦门天气怎么样？", # 复合意图，看它怎么归
]
for q in questions:
    intent, ans = workflow(q)
    print(f"[路由结果] {intent}")
    print(f"最终答案：\n{merge_answers(q, intent, ans)}\n{'-'*70}")
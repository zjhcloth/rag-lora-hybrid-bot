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
            {"role": "system", "content": f"""你只负责回答公司制度相关的问题（年假、报销、远程办公等）。
如果用户问题中包含非制度类内容（如天气、闲聊），你只回答制度部分，
对其余部分明确说“这部分不属于我的职责范围”。

根据以下资料回答。资料中没有的信息说“资料未提及”，不要编造。
资料：
{context}"""},
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
    messages = [
        {"role": "system", "content": "你只负责天气查询。如果问题中包含非天气内容，只回答天气部分，其余说“不属于我的职责范围”。"},
        {"role": "user", "content": question}
    ]
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
    routed = route(question)                 # {"tasks": [{"intent", "query"}, ...]}
    tasks = routed["tasks"]
    answers = {}
    if len(tasks) == 1:                      # 单任务不开线程池
        t = tasks[0]
        answers[0] = NODE_MAP[t["intent"]](t["query"])
    else:                                    # 多任务并行执行
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(lambda t: NODE_MAP[t["intent"]](t["query"]), tasks))
            answers = {i: r for i, r in enumerate(results)}
#这样写的顺序是不对的，因为 map 是并行的，所以 answers 可能不是按顺序的
# as_completed 返回的是完成顺序，不是 task 顺序。所以 answers.items() 遍历时，blocks 的顺序是“谁先跑完谁在前”。
# 这次天气快、rag 慢，blocks 就是【weather】在前；下次网络抖动 rag 快，顺序就反过来。merge 的输入顺序每次可能不同，
# 输出就不可复现——你做第七关评测时会发现“同一个问题跑两次，merge 后措辞有差异”，很难判断是模型抖动还是系统 bug。
#        with concurrent.futures.ThreadPoolExecutor() as ex:
#            futures = {ex.submit(NODE_MAP[t["intent"]], t["query"]): i for i, t in enumerate(tasks)}
#            for f in concurrent.futures.as_completed(futures):
#                answers[futures[f]] = f.result()
    return [t["intent"] for t in tasks], answers   # 统一返回 (intent 列表, {下标: 答案})

# ========== 测试：每个分支至少一条 + 一条复合问题 ==========
questions = [
    "单笔 6000 元的报销需要谁审批？",     # 应走 rag
    "厦门今天天气怎么样？",               # 应走 weather
    "讲个笑话吧",                         # 应走 chat
    "公司年假多少天，另外厦门天气怎么样？", # 复合意图，看它怎么归
]
for q in questions:
    intent, ans = workflow(q)
    final = merge_answers(q, ans) if len(ans) > 1 else ans[0]
    print(f"[路由] {intent}")
    print(f"最终答案：\n{final}\n{'-'*70}")
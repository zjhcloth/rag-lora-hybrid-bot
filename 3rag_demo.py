import os, glob
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# ---------- 1. 读取并分段 ----------
chunks = []
for path in glob.glob("docs/*.txt"):
    text = open(path, encoding="utf-8").read()
    # 按空行切段，再按长度合并到 200~500 字
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""
    for p in paras:
        if len(buf) + len(p) < 400:
            buf += p + "\n"
        else:
            if buf: chunks.append(buf.strip())
            buf = p + "\n"
    if buf: chunks.append(buf.strip())

print(f"共 {len(chunks)} 个片段")
for i, c in enumerate(chunks):
    print(f"[{i}] {c[:40]}...")

# ---------- 2. 向量化 + 入库 ----------
embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = embedder.encode(chunks, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])   # 归一化后内积 = 余弦相似度
index.add(emb)

# ---------- 3. LLM 客户端 ----------
client = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)

def ask(question, use_rag=True, k=3):
    if use_rag:
        q_emb = embedder.encode([question], normalize_embeddings=True)
        D, I = index.search(q_emb, k)
        print("命中片段:", I[0], "相似度:", D[0])   # 临时调试用
        context = "\n\n".join(chunks[i] for i in I[0])
        system = f"根据以下资料回答问题。资料中没有的信息就说不知道，不要编造。\n\n资料：\n{context}"
    else:
        system = "你是一个助手，请回答用户问题。"
    
    resp = client.chat.completions.create(
        model="glm-5.3-flash",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )
    return resp.choices[0].message.content

# ---------- 4. 对比实验 ----------
questions = [
    "星辰科技正式员工入职满三年有多少天年假？",
    "单笔 6000 元的报销需要谁审批？",
    "新员工试用期可以远程办公吗？",
]

for q in questions:
    print("=" * 70)
    print("问题：", q)
    print("--- 无 RAG ---")
    print(ask(q, use_rag=False))
    print("--- 有 RAG ---")
    print(ask(q, use_rag=True))
    print()
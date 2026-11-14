# 8serve.py
# 启动：uvicorn 8serve:app --host 0.0.0.0 --port 8000
# 测试：curl http://localhost:8000/health
#       curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"question":"年假多少天？"}'

import os, json, glob, time, logging, threading, traceback
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
#API_MODEL = "glm-5.3-flash"
API_MODEL = "glm-4-flash"
DOCS_DIR = "docs"

PREFIX = "您好，星辰科技为您解答："
SUFFIX = "还有什么可以帮您？"
REJECT_HINTS = ["不在我的知识范围内", "资料中没有", "资料未提及"]
FALLBACK_ANSWER = f"{PREFIX}系统暂时繁忙，请稍后再试。{SUFFIX}"

# ============================================================
# 日志：stdout，带时间戳，方便 tail -f
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("serve")

# ============================================================
# 全局状态（启动时填充一次）
# ============================================================
state = {
    "ready": False,
    "ft_model": None,
    "tok": None,
    "embedder": None,
    "index": None,
    "chunks": None,
    "llm": None,
    "load_error": None,
    "n_chunks": 0,
}
ft_lock = threading.Lock()   # 微调模型推理串行化，避免并发抢占

# ============================================================
# 加载：RAG 库 + 微调模型
# ============================================================
def build_rag():
    chunks = []
    for path in glob.glob(f"{DOCS_DIR}/*.txt"):
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
    if not chunks:
        raise RuntimeError(f"{DOCS_DIR}/ 下没有可用 txt 文档")
    embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    emb = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb)
    state.update(chunks=chunks, embedder=embedder, index=index, n_chunks=len(chunks))
    log.info(f"RAG 库就绪：{len(chunks)} 个片段")

def build_ft():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
    lora_model = PeftModel.from_pretrained(base, ADAPTER)
    ft = lora_model.merge_and_unload()
    ft = ft.to("mps")
    ft.eval()
    state.update(ft_model=ft, tok=tok)
    log.info(f"微调模型就绪：{ADAPTER}，设备={next(ft.parameters()).device}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    try:
        state["llm"] = OpenAI(
            api_key=os.environ.get("ZCODE_API_KEY"),
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        build_rag()
        build_ft()
        state["ready"] = True
        log.info(f"全部就绪，耗时 {time.time() - t0:.1f}s")
    except Exception as e:
        state["load_error"] = f"{type(e).__name__}: {e}"
        log.error(f"启动失败：{state['load_error']}\n{traceback.format_exc()}")
    yield
    log.info("服务关闭")

app = FastAPI(title="星辰科技客服", lifespan=lifespan)

# ============================================================
# 核心逻辑
# ============================================================
def is_reject(ans: str) -> bool:
    return any(h in ans for h in REJECT_HINTS)

def ask_ft(q: str) -> str:
    t0 = time.time()
    tok, ft = state["tok"], state["ft_model"]
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )
    inputs = {k: v.to("mps") for k, v in inputs.items()}
    with ft_lock, torch.no_grad():
        out = ft.generate(**inputs, max_new_tokens=80, do_sample=False)
    t1 = time.time()
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    log.info(f"ft 本地推理: {time.time()-t0:.2f}s, 输出长度={new_tokens.shape[0]}, 设备={next(ft.parameters()).device}")
    return tok.decode(new_tokens.cpu(), skip_special_tokens=True).strip().cpu()

def retrieve(q, k=3):
    """只做检索，不调 LLM"""
    q_emb = state["embedder"].encode([q], normalize_embeddings=True)
    D, I = state["index"].search(q_emb, k)
    kept = [(float(D[0][j]), int(I[0][j])) for j in range(len(I[0])) if float(D[0][j]) > 0.5]
    if not kept:
        kept = [(float(D[0][0]), int(I[0][0]))]
    context = "\n\n".join(state["chunks"][idx] for _, idx in kept)
    return context, kept[0][0]

def generate(q, context):
    """只调 LLM，不含检索"""
    resp = state["llm"].chat.completions.create(
        model=API_MODEL, temperature=0.1, max_tokens=120,
        messages=[
            {"role": "system", "content": f"""你是星辰科技客服，只回答公司制度相关问题。

资料：
{context}

输出要求：
- 必须以"{PREFIX}"开头
- 必须以"{SUFFIX}"结尾
- 资料中没有相关信息时，正文必须写"该问题暂不在我的知识范围内，建议您联系 HR 确认"，不得用通用知识回答"""},
            {"role": "user", "content": q},
        ],
    )
    return resp.choices[0].message.content.strip()

# 固定拒绝话术（不依赖 LLM，风格可控）
LOW_SIM_REJECT = f"{PREFIX}该问题暂不在我的知识范围内，建议您联系 HR 确认。{SUFFIX}"

def chat(q):
    try:
        context, top1 = retrieve(q)
    except Exception as e:
        log.error(f"检索异常：{e}")
        return FALLBACK_ANSWER, "fallback", True

    log.info(f"top1={top1:.3f}")
    # 核心改动：相似度低时直接拒绝，不调 LLM
    if top1 < 0.48:
        return LOW_SIM_REJECT, "rag(拒绝·低相似)", True

    try:
        rag_ans = generate(q, context)
    except Exception as e:
        log.error(f"LLM 异常：{e}")
        return FALLBACK_ANSWER, "fallback", True

    if not is_reject(rag_ans):
        return rag_ans, "rag", False

    try:
        ft_ans = ask_ft(q)
    except Exception as e:
        log.error(f"微调异常：{e}")
        ft_ans = FALLBACK_ANSWER
    if is_reject(ft_ans):
        return rag_ans, "rag(拒绝)", True
    return ft_ans, "rag→ft", False

# ============================================================
# 请求 / 响应模型
# ============================================================
class ChatReq(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)

class ChatResp(BaseModel):
    answer: str
    path: str
    rejected: bool
    elapsed_ms: int

# ============================================================
# 路由
# ============================================================
@app.get("/health")
def health():
    body = {
        "ready": state["ready"],
        "n_chunks": state["n_chunks"],
        "adapter": ADAPTER,
        "load_error": state["load_error"],
    }
    if not state["ready"]:
        raise HTTPException(status_code=503, detail=body)
    return body

@app.post("/chat", response_model=ChatResp)
def chat_endpoint(req: ChatReq):
    if not state["ready"]:
        raise HTTPException(status_code=503, detail="服务未就绪")
    t0 = time.time()
    q = req.question.strip()
    try:
        answer, path, rejected = chat(q)
    except Exception as e:
        log.error(f"未捕获异常：{type(e).__name__}: {e}\n{traceback.format_exc()}")
        answer, path, rejected = FALLBACK_ANSWER, "fallback", True
    elapsed = int((time.time() - t0) * 1000)
    log.info(f"chat | q={q[:30]!r} | path={path} | rejected={rejected} | {elapsed}ms")
    return ChatResp(answer=answer, path=path, rejected=rejected, elapsed_ms=elapsed)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("8server:app", host="0.0.0.0", port=8000, reload=False)

import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)

# ---------- 1. 定义真实函数（mock 数据）----------
def get_weather(city: str) -> dict:
    fake = {"厦门": {"temp": 26, "desc": "多云"}, "北京": {"temp": 8, "desc": "晴"}}
    return fake.get(city, {"temp": None, "desc": "未知城市"})

# ---------- 2. 用 JSON Schema 向模型描述工具 ----------
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的当前天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 厦门"}},
            "required": ["city"]
        }
    }
}]

def run_agent(user_msg: str):
    messages = [{"role": "user", "content": user_msg}]
    
    # 第一轮：模型决定是否调用工具
    resp = client.chat.completions.create(
        model="glm-5.3-flash",
        messages=messages,
        tools=tools,
        temperature=0
    )
    msg = resp.choices[0].message
    print("第一轮模型返回：", msg.tool_calls if msg.tool_calls else msg.content)
    
    if not msg.tool_calls:
        return msg.content   # 模型认为不需要工具，直接答
    
    # 执行工具，结果回传
    messages.append(msg)   # 把带 tool_calls 的 assistant 消息放回历史
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_weather(**args)
        print(f"  执行 {call.function.name}({args}) -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })
    
    # 第二轮：模型基于工具结果生成最终答案
    final = client.chat.completions.create(
        model="glm-5.3-flash",
        messages=messages,
        tools=tools,
        temperature=0
    )
    return final.choices[0].message.content

# ---------- 3. 测试三类输入 ----------
for q in ["厦门今天天气怎么样？", "北京和厦门哪里更热？", "你好，讲个笑话"]:
    print("=" * 60)
    print("用户：", q)
    print("最终：", run_agent(q))
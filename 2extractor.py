import os
from openai import OpenAI

client = OpenAI(
    # 1. 填入你在 ZCode / 对应平台生成的 API Key（不是 DeepSeek 的 Key）
    api_key=os.environ.get("ZCODE_API_KEY"),
    
    # 2. 填入 Coding Plan 专用端点，而不是通用端点
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"  # BigModel 示例，Z.ai 需替换
)
for temp in [0.1,0.9]:
    for i in range(3):
        response = client.chat.completions.create(
            # 3. 换成 ZCode 套餐支持的模型 ID，例如 GLM 系列
            model="glm-5.3-flash",  # 或 glm-5.3 等
    
            # messages=[{"role": "user", "content": "Hello!"}]
            messages=[
                {"role": "system", "content": "你是一个简洁的助手。"},
                {"role": "user", "content": "用一句话解释RAG。"}
            ],
            temperature=temp
        )
        print(f"--- temperature={temp} 第{i+1}次 ---")
        print(response.choices[0].message.content)
        print("usage:",response.usage)
        print()

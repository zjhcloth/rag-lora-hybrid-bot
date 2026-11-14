import os, json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("ZCODE_API_KEY"),
    base_url="https://open.bigmodel.cn/api/coding/paas/v4"
)

# ---------- 5 条测试输入（难度递增）----------
test_inputs = [
    "《三体》是刘慈欣于2008年由重庆出版社出版的长篇科幻小说。",
    "《活着》这本书是余华写的。",
    "推荐一本好书：《人类简史》，尤瓦尔·赫拉利著，2014年中信出版社出版，历史类。另外我最近也在看《三体》。",
    "《代码大全》第二版，Steve McConnell 著，2006 年电子工业出版社，属于软件工程领域。",
    "这本书叫《小王子》，作者是法国的圣埃克苏佩里，1943年首次出版。",
]

# ---------- 3 种配置 ----------
bare_prompt = "请从下面文本中提取书籍信息，输出 JSON，字段为 title/author/year/publisher/genre。"

fewshot_system = """请从下面文本中提取书籍信息，输出 JSON，字段为 title/author/year/publisher/genre。
year 必须是数字类型（整数），缺失字段填 null，只输出 JSON，不要加任何解释或 markdown 代码块。

示例1：
输入：刘慈欣的《三体》于2008年由重庆出版社出版，是科幻小说。
输出：{"title": "三体", "author": "刘慈欣", "year": 2008, "publisher": "重庆出版社", "genre": "科幻"}

示例2：
输入：《活着》是余华的作品。
输出：{"title": "活着", "author": "余华", "year": null, "publisher": null, "genre": null}"""

configs = [
    ("裸Prompt",       bare_prompt,    0.9),
    ("Few-shot",       fewshot_system, 0.9),
    ("Few-shot+低温",  fewshot_system, 0.1),
]

# ---------- 双重循环跑完所有组合 ----------
for name, system, temp in configs:
    print("=" * 70)
    print(f"配置：{name} | temperature={temp}")
    print("=" * 70)
    for idx, text in enumerate(test_inputs, 1):
        response = client.chat.completions.create(
            model="glm-5.3-flash",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
            temperature=temp
        )
        out = response.choices[0].message.content.strip()

        # 自动校验：能否被 json.loads 解析
        try:
            json.loads(out)
            check = "✅ 合法 JSON"
        except json.JSONDecodeError as e:
            check = f"❌ 非法 JSON（{e.msg}）"

        print(f"[输入{idx}] {text[:32]}...")
        print(f"  输出：{out}")
        print(f"  校验：{check}")
        print()

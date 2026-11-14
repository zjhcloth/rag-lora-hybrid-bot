#!/bin/bash
API_KEY="$ZCODE_API_KEY"
URL="https://open.bigmodel.cn/api/paas/v4/chat/completions"

SHORT='{"model":"glm-4-flash","messages":[{"role":"user","content":"你好"}]}'
LONG=$(cat <<'EOF'
{"model":"glm-4-flash","max_tokens":120,"temperature":0.1,"messages":[{"role":"system","content":"你是星辰科技客服，只回答公司制度相关问题。\n资料：年假 15 天、满五年 20 天；报销每月 25 日截止、超 5000 元需总监二次审批；远程每周最多 2 天、试用期不适用。\n输出必须以您好，星辰科技为您解答：开头，以还有什么可以帮您？结尾；不属于公司制度范畴或资料没有的问题，正文写该问题暂不在我的知识范围内，建议您联系 HR 确认。"},{"role":"user","content":"公司有股票期权吗？"}]}
EOF
)

echo "=== 短请求 ×3 ==="
for i in 1 2 3; do
  curl -s -o /dev/null -w "  #$i  connect=%{time_connect}s  tls=%{time_appconnect}s  首字=%{time_starttransfer}s  总=%{time_total}s\n" \
    -X POST "$URL" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$SHORT"
done

echo ""
echo "=== 长请求 ×3 ==="
for i in 1 2 3; do
  curl -s -o /dev/null -w "  #$i  connect=%{time_connect}s  tls=%{time_appconnect}s  首字=%{time_starttransfer}s  总=%{time_total}s\n" \
    -X POST "$URL" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" -d "$LONG"
done
http://127.0.0.1:8000
ngrok http 8000


【浏览器 / 前端】
   |
   | 1. 用户输入一句话
   |    POST /api/chat
   |    （前端主动发起 HTTP 请求）
   v
【FastAPI 后端】
   |
   | 2. FastAPI 接收到用户请求
   |    解析 sender / message
   |
   | 3. FastAPI 作为 HTTP 客户端
   |    向 Rasa 发送请求
   |    POST http://localhost:5005/webhooks/rest/webhook
   |    （FastAPI 主动）
   v
【Rasa Server】
   |
   | 4. Rasa 在“同一个 HTTP 请求中”完成处理：
   |    - NLU（意图识别、实体抽取）
   |    - Core（对话状态 & 策略）
   |    - 可选：调用 Action Server（5055）
   |
   | 5. Rasa 生成回复内容
   |    并作为 HTTP Response 返回
   |    （不是新请求）
   v
【FastAPI 后端】
   |
   | 6. FastAPI 接收 Rasa 的 HTTP 响应
   |    解析 JSON（可能多条消息）
   |    整理 / 转换为前端需要的结构
   |
   | 7. FastAPI 返回 HTTP Response
   |    给前端
   v
【浏览器 / 前端】
   |
   | 8. 前端接收返回数据
   |    渲染到聊天窗口


方案一：投诉严重程度判断（更合理） 比如： 用户输入： “有点慢” “我三天没收到” “我要投诉到消协” “我要告你们” 你可以微调 BERT 做： 轻度投诉 中度投诉 强烈投诉
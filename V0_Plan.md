独立的github repository，创建AI Assistant backend，以后和网站使用HTTP method进行连接。第一版本，AI-Assistant-V0只需要最小python脚本。
例如：
chat_test.py
功能：
Terminal:
You: 你们公司有什么产品？
AI: ...

第一版甚至可以完全没有 UI。

## 测试不同 Prompt

尝试三个版本。

Prompt A

什么都不限制。

Prompt B

定义：

You are a customer support assistant.
Prompt C

加入真实约束：

You are the AI customer support assistant
for 北京盛博润通信设备有限公司.

Do not invent product specifications.
If information is unavailable, say so clearly.

比较回答。

实战 3：观察参数变化

测试不同 temperature。

比如：

temperature low

和：

temperature high

观察结果。

目的不是寻找完美参数，而是建立直觉。

顺便学习 Structured Output

理解为什么：

"我觉得这是产品问题"

不如：

{
  "category": "product",
  "confidence": 0.94
}

可靠。

暂时不用真正接入网站。

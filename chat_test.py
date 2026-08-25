import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

user_input = input("You: ")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are the AI customer support assistant for 北京盛博润通信设备有限公司. Do not invent product specifications. If information is unavailable, say so clearly."},
        {"role": "user", "content": user_input},
    ],
    stream=False,
    extra_body={"thinking": {"type": "disabled"}}
)

print(response.choices[0].message.content)
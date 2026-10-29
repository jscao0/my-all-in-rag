from openai import OpenAI
import os

from openai.types.responses import response

client = OpenAI(
    api_key=os.environ['DEEPSEEK_API_KEY'],
    base_url=os.environ['DEEPSEEK_BASE_URL'],
)

def send_messages(messages, tools=None):
    response = client.chat.completions.create(
        model='deepseek_chat',
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"]
            },
        }
    },
]


messages = [{
    "role":"user",
    "content":"今天北京天气怎么样"
}]
print(f"User> {messages[0]['content']}\n")
message = send_messages(messages, tools)

if message.tool_calls:
    print("--- 模型发起了工具调用 ---")
    tool_call = message.tool_calls[0]
    function_info = tool_call.function
    print(f"工具名称: {function_info.name}")
    print(f"工具参数: {function_info.arguments}")


    messages.append(message)


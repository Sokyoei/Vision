"""
main
"""

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 初始化聊天对象
chat = ChatOpenAI(openai_api_key="...")

# 向聊天模型发问
batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming."),
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence."),
    ],
]
result = chat.generate(batch_messages)

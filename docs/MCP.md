# MCP

MCP(Model Context Protocol)「模型上下文协议」。

2024年11月底，由 Anthropic 推出的一种开放标准，旨在统一大型语言模型(LLM)与外部数据源和工具之间的通信协议。
MCP 的主要目的在于解决当前 AI 模型因数据孤岛限制而无法充分发挥潜力的难题，MCP 使得 AI 应用能够安全地访问和操作本地及远程数据，
为 AI 应用提供了连接万物的接口。

Function Calling 是 AI 模型调用函数的机制，MCP 是一个标准协议，使 AI 模型与 API 无缝交互，而 AI Agent 是一个自主运行的智能
系统，利用 Function Calling 和 MCP 来分析和执行任务，实现特定目标。

## MCP 核心架构

- MCP 主机(MCP Hosts): 发起请求的 LLM 应用程序(例如 Claude Desktop、IDE 或 AI 工具)。
- MCP 客户端(MCP Clients): 在主机程序内部，与 MCP server 保持 1:1 的连接。
- MCP 服务器(MCP Servers): 为 MCP client 提供上下文、工具和 prompt 信息。
- 本地资源(Local Resources): 本地计算机中可供 MCP server 安全访问的资源(例如文件、数据库)。
- 远程资源(Remote Resources): MCP server 可以连接到的远程资源(例如通过 API)。

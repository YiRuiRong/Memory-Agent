# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个扫地机器人智能客服 Agent 项目，基于 LangChain `create_agent`（ReAct 模式）构建，前端为 Streamlit。使用阿里云 DashScope 的 Tongyi (`qwen3-max`) 作为对话模型、`text-embedding-v4` 作为向量模型，Chroma 作为向量库。

运行前需配置环境变量 `DASHSCOPE_API_KEY`（`langchain_community` 的 `ChatTongyi` / `DashScopeEmbeddings` 需要）。

## 常用命令

```bash
# 启动 Streamlit 前端（主入口）
streamlit run app.py

# 加载 data/ 目录下的知识文件到 Chroma 向量库（首次运行或新增资料后执行）
# 该脚本依赖 utils.path_tool 计算绝对路径，必须以模块方式执行，不能直接 python rag/vector_store.py
python -m rag.vector_store

# 单独调试 Agent（命令行流式输出）
python -m agent.react_agent

# 单独调试 RAG 总结链
python -m rag.rag_service
```

项目内任何带 `from utils.xxx import` 的文件都必须从项目根目录以 `python -m <package>.<module>` 方式运行，否则相对路径与包导入都会失败。

## 架构要点

### 顶层数据流
`app.py` → `ReactAgent.execute_stream()` → `langchain.agents.create_agent` 构建的 Agent → 工具链 / RAG → 响应流。

### Agent 组装（`agent/react_agent.py`）
`create_agent` 注册了三类组件：
- **tools**：定义于 `agent/tools/agent_tools.py`，包括 `rag_summarize`、`get_weather`（mock）、`get_user_location`（mock，随机城市）、`get_user_id`（mock，随机 1001-1010）、`get_current_month`（mock，随机 2025-XX）、`fetch_external_data`、`fill_context_for_report`。
- **middleware**：定义于 `agent/tools/middleware.py`，有 `monitor_tool`（`@wrap_tool_call`）、`log_before_model`（`@before_model`）、`report_prompt_switch`（`@dynamic_prompt`）。
- **system_prompt**：由 `utils.prompt_loader.load_system_prompts()` 从 `prompts/main_prompt.txt` 读取。

### 动态提示词切换（跨多文件的关键机制）
这是整个 Agent 最核心的非显式行为，理解它需要同时看三处代码：

1. `agent/react_agent.py` 在 `agent.stream(..., context={"report": False})` 传入初始 runtime context。
2. 用户触发报告生成场景时，主提示词 (`prompts/main_prompt.txt`) 指示模型先调用 `fill_context_for_report` 工具。
3. `agent/tools/middleware.py` 的 `monitor_tool` 拦截该工具调用，把 `request.runtime.context["report"] = True`。
4. 下一轮模型调用前，`report_prompt_switch` (`@dynamic_prompt`) 检测到该标志，改用 `prompts/report_prompt.txt` 作为系统提示词。

因此 `fill_context_for_report` 工具本身没有业务逻辑，它是"信号工具"——修改它或更换其名字必须同步修改 `monitor_tool` 中的字符串判断以及两个提示词文件中的调用指令。

### RAG 管线（`rag/`）
- `VectorStoreService` 封装 Chroma 客户端与 `RecursiveCharacterTextSplitter`。`load_document()` 会扫描 `data/` 下允许后缀的文件，对每个文件计算 MD5，**以 `md5.text` 为去重账本**（纯文本，每行一个 md5 十六进制）跳过已入库的文件。删除或改坏 `md5.text` 会导致重复入库。
- `RagSummarizeService` 的链：`PromptTemplate → print_prompt(打印调试) → chat_model → StrOutputParser`。被包成 `rag_summarize` 工具供 Agent 使用。

### 模型工厂（`model/factory.py`）
用简单工厂模式生成全局单例 `chat_model` 和 `embed_model`；切换模型只需改 `config/rag.yml`。

### 配置与路径
- 四个 YAML 在 `config/`：`agent.yml`（外部数据文件路径）、`chroma.yml`（向量库配置、分片、md5 账本位置、允许后缀）、`prompts.yml`（三个提示词文件路径）、`rag.yml`（模型名）。
- 加载入口是 `utils/config_handler.py`，在 import 时即读入四个全局 dict：`rag_conf` / `chroma_conf` / `prompts_conf` / `agent_conf`。
- **所有路径配置都是相对项目根目录**，读取时必须走 `utils.path_tool.get_abs_path()` 转绝对路径（`path_tool` 通过 `__file__` 向上两级推断项目根）。不要硬编码路径或改变 `utils/` 目录层级。

### 日志（`utils/logger_handler.py`）
全局共享 `logger`（name=`agent`），控制台 INFO + 文件 DEBUG，文件按日期切分写入 `logs/agent_YYYYMMDD.log`。模块被 import 时自动创建 `logs/`。

## 数据与环境注意事项

- `data/external/records.csv` 被 `fetch_external_data` 工具按自定义格式（引号包裹、逗号分隔、首行跳过）解析到内存字典；列顺序固定为 `user_id, 特征, 效率, 耗材, 对比, 月份`，修改列顺序需同步改 `agent_tools.py::generate_external_data`。
- `chroma_db/` 是 Chroma 持久化目录，项目根与 `agent/`、`rag/` 下各有一个同名目录——**真正生效的是根目录下的 `chroma_db/`**（`chroma.yml` 配的相对路径，经 `get_abs_path` 解析）；其它是遗留或不同工作目录下运行产生的，正常情况下可忽略。

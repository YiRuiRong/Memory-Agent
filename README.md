# 扫地机器人智能客服 Agent

这是一个基于大模型、RAG 知识库和工具调用的扫地机器人智能客服项目。用户可以通过 Streamlit 页面提问，Agent 会根据问题自主调用知识库检索、天气查询、用户信息查询、外部使用记录查询等工具，并生成最终回答。

项目核心技术栈：

- LangChain `create_agent`，ReAct Agent 模式
- Chroma，向量库和知识库检索
- 阿里云 DashScope 通义千问，默认对话模型 `qwen3-max`
- DashScope Embedding，默认向量模型 `text-embedding-v4`
- Streamlit，Web 聊天界面

## 功能特性

- 智能客服问答：基于 `data/` 中的扫地机器人知识资料回答售后、维护、故障排除、选购等问题。
- RAG 检索增强：先检索 Chroma 向量库，再让大模型基于参考资料总结回答。
- 多工具调用：支持天气、用户位置、用户 ID、月份、外部业务记录等工具。
- 动态提示词切换：当用户要求生成使用报告时，Agent 会通过信号工具切换到报告生成提示词。
- 长期记忆机制：项目内置 `memory/` 目录和记忆系统，用于保存稳定事实、SOP 和会话归档。
- 日志记录：运行日志自动写入 `logs/agent_YYYYMMDD.log`。

## 项目结构

```text
.
├── app.py                      # Streamlit Web 入口
├── agent/
│   ├── react_agent.py          # Agent 组装和流式执行
│   ├── memory_system.py        # L1/L2/L3/L4 记忆系统
│   └── tools/
│       ├── agent_tools.py      # Agent 工具定义
│       └── middleware.py       # 工具监控、模型前日志、动态提示词切换
├── rag/
│   ├── vector_store.py         # 文档入库、切片、Chroma 持久化
│   └── rag_service.py          # RAG 检索和总结链
├── model/
│   └── factory.py              # 对话模型和向量模型工厂
├── prompts/
│   ├── main_prompt.txt         # 默认客服提示词
│   ├── report_prompt.txt       # 使用报告提示词
│   └── rag_summarize.txt       # RAG 总结提示词
├── config/
│   ├── agent.yml               # 外部业务数据配置
│   ├── chroma.yml              # 向量库、切片、检索配置
│   ├── prompts.yml             # 提示词路径配置
│   └── rag.yml                 # 模型名称配置
├── data/                       # 知识库原始资料
│   └── external/records.csv    # 模拟外部业务系统数据
├── memory/                     # Agent 记忆文件
├── docs/                       # 项目说明文档
├── requirements.txt            # Python 依赖
└── .gitignore                  # Git 忽略规则
```

运行后会生成以下本地文件或目录，这些不需要提交到 GitHub：

- `chroma_db/`：Chroma 向量库持久化目录
- `md5.text`：知识库文件入库去重记录
- `logs/`：运行日志
- `memory/L4_raw_sessions/`：会话归档
- `__pycache__/`：Python 缓存

## 环境要求

建议使用：

- Python 3.11
- Windows、macOS 或 Linux
- 阿里云 DashScope API Key

本项目依赖 DashScope 的 `ChatTongyi` 和 `DashScopeEmbeddings`，运行前必须配置环境变量 `DASHSCOPE_API_KEY`。

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd AI大模型RAG与智能体开发_Agent项目
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS 或 Linux：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 DashScope API Key

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope API Key"
```

macOS 或 Linux：

```bash
export DASHSCOPE_API_KEY="你的 DashScope API Key"
```

如果希望长期生效，可以把环境变量配置到系统环境变量、Shell profile 或 `.env` 管理工具中。注意不要把真实 API Key 提交到 GitHub。

### 5. 构建知识库

第一次运行，或新增、修改 `data/` 中的知识资料后，需要执行：

```bash
python -m rag.vector_store
```

注意：请在项目根目录执行，并使用 `python -m` 模块方式运行。不要直接执行 `python rag/vector_store.py`，否则可能出现包导入或路径解析问题。

### 6. 启动 Web 应用

```bash
streamlit run app.py
```

启动后浏览器访问：

```text
http://localhost:8501
```

## 常用命令

```bash
# 构建或更新知识库
python -m rag.vector_store

# 启动 Streamlit 页面
streamlit run app.py

# 命令行调试 Agent
python -m agent.react_agent

# 单独调试 RAG 服务
python -m rag.rag_service
```

## 典型使用场景

### 售后咨询

用户输入：

```text
我家机器人老是撞墙怎么办？
```

Agent 会调用 `rag_summarize`，从 `data/` 中检索故障排除资料，然后总结回答。

### 环境建议

用户输入：

```text
我在杭州，今天适合开扫地机吗？
```

Agent 可以调用位置和天气工具，结合环境信息给出建议。

### 使用报告生成

用户输入：

```text
给我生成我的使用报告
```

Agent 会触发报告生成流程，切换到 `prompts/report_prompt.txt`，并结合 `data/external/records.csv` 中的模拟业务数据生成 Markdown 报告。

## 核心机制

### 记忆系统四层架构

记忆系统主要由 `agent/memory_system.py` 实现。代码中定义了 L1/L2/L3/L4 四层长期记忆，同时还存在一个辅助文件 `memory/memory_management_sop.md`。这个辅助文件在代码里对应 `L0_PATH`，用于记录记忆管理规则，但它不是四层记忆之一。

四层记忆的真实路径和职责如下：

| 层级 | 文件或目录 | 代码常量 | 实际职责 |
| --- | --- | --- | --- |
| L1 | `memory/global_mem_insight.txt` | `L1_PATH` | 短小的全局记忆索引，会通过 `get_global_memory()` 常驻追加到 system prompt。它不适合存放大段事实，主要用于指向 L2/L3 中更详细的内容。 |
| L2 | `memory/global_mem.txt` | `L2_PATH` | 已验证、相对稳定的长期事实库，例如项目事实、配置、路径、稳定约束、用户明确偏好。默认不会全文注入模型，只通过 L1 提供索引提示。 |
| L3 | `memory/sop_library.md` | `L3_PATH` | 可复用 SOP、流程、排查方法和集成说明。适合保存“以后遇到同类任务怎么做”的过程性知识。 |
| L4 | `memory/L4_raw_sessions/` | `L4_DIR` | 会话与日志归档目录。当前会话写入 `current_session.log`，归档结果追加到 `all_histories.txt`，默认不会注入模型上下文。 |

初始化发生在 `ensure_memory_files()`：它会创建 `memory/`、`memory/L4_raw_sessions/`，并在文件缺失时写入 L1/L2/L3/L0 的默认内容，同时创建 `file_access_stats.json`、`current_session.log` 和 `all_histories.txt`。

运行时还有一层进程内工作记忆，由 `AgentMemoryRuntime` 维护，不直接等同于 L1-L4。它保存当前轮次的 `history_info`、`key_info`、`related_sop`、工具结果和候选记忆。每次模型调用前，`agent/tools/middleware.py` 中的 `log_before_model` 会调用 `build_anchor_state_update()`，把一段 `<runtime_memory>` 锚点插入消息列表。这段锚点包含最近会话摘要、当前轮次、短期 checkpoint 和相关 SOP 提示。

长期记忆沉淀链路如下：

```text
用户输入
    -> ReactAgent.execute_stream()
    -> memory.record_user_turn()
    -> 工具调用被 monitor_tool 捕获
    -> memory.record_tool_result()
    -> 本轮结束后 turn_end_callback()
    -> collect_memory_candidates()
    -> should_distill_memory()
    -> start_long_term_update()
    -> classify_memory_candidate()
    -> apply_memory_update()
    -> 写入 L1/L2/L3，或拒绝写入
```

候选记忆不会无条件写入。`classify_memory_candidate()` 会先过滤疑似 secret、token、密码、密钥，以及天气、位置、用户 ID、月份、外部记录等易变工具结果。通过校验后，内容才会按规则进入 L1、L2 或 L3；如果写入 L2/L3 成功，代码还会自动向 L1 写入一条指向 `memory/global_mem.txt` 或 `memory/sop_library.md` 的索引。

L4 归档由 `archive_sessions()` 处理。它会扫描 `memory/L4_raw_sessions/current_session.log` 和 `logs/*.log` 中的 `[USER]`、`[Agent]` 摘要行，把符合条件的历史追加到 `memory/L4_raw_sessions/all_histories.txt`。默认会跳过最近 2 小时内仍在写入的日志，除非当前会话被 `finalize_session_log()` 标记完成，或 `current_session.log` 超过 `CURRENT_SESSION_MAX_BYTES`。

### RAG 知识库流程

知识库构建流程：

```text
data/*.txt 或 data/*.pdf
    -> 读取文档
    -> 计算 MD5
    -> 根据 md5.text 判断是否已入库
    -> 文档切片
    -> DashScope Embedding 向量化
    -> 写入 Chroma
```

问答流程：

```text
用户问题
    -> Agent 调用 rag_summarize
    -> Chroma 检索 top-k 文档
    -> 拼接参考资料
    -> 大模型总结
    -> 返回答案
```

### 动态提示词切换

默认情况下，Agent 使用 `prompts/main_prompt.txt` 作为客服提示词。

当用户要求生成使用报告时，模型会先调用 `fill_context_for_report` 工具。该工具本身不处理业务数据，它是一个信号工具。`agent/tools/middleware.py` 中的 `monitor_tool` 会拦截这个工具调用，并把运行时上下文中的 `report` 标记设为 `True`。

下一轮模型调用时，`report_prompt_switch` 会检测该标记，并把系统提示词切换为 `prompts/report_prompt.txt`。

相关文件：

- `agent/react_agent.py`
- `agent/tools/agent_tools.py`
- `agent/tools/middleware.py`
- `prompts/main_prompt.txt`
- `prompts/report_prompt.txt`

## 配置说明

| 需求 | 修改文件 |
| --- | --- |
| 更换对话模型或向量模型 | `config/rag.yml` |
| 修改 Chroma 目录、检索 top-k、切片大小 | `config/chroma.yml` |
| 修改提示词文件路径 | `config/prompts.yml` |
| 修改外部业务数据路径 | `config/agent.yml` |
| 修改客服人设和工具调用规则 | `prompts/main_prompt.txt` |
| 修改报告生成格式 | `prompts/report_prompt.txt` |
| 修改 RAG 总结约束 | `prompts/rag_summarize.txt` |

所有配置路径都按项目根目录解析，底层通过 `utils.path_tool.get_abs_path()` 转成绝对路径。

## 知识库重建

如果只新增了文档，直接重新执行：

```bash
python -m rag.vector_store
```

如果修改了已有文档，但发现结果没有更新，可以删除 `md5.text` 后重新入库：

```bash
python -m rag.vector_store
```

如果想彻底重建向量库，需要同时删除：

```text
chroma_db/
md5.text
```

然后重新执行：

```bash
python -m rag.vector_store
```

## 常见问题

### 1. 报 `ModuleNotFoundError: utils`

请确认你在项目根目录运行命令，并使用模块方式：

```bash
python -m rag.vector_store
```

不要使用：

```bash
python rag/vector_store.py
```

### 2. 报 `DASHSCOPE_API_KEY` 缺失

请先配置环境变量：

```powershell
$env:DASHSCOPE_API_KEY="你的 DashScope API Key"
```

### 3. RAG 没有返回有效资料

检查是否已经执行过：

```bash
python -m rag.vector_store
```

同时确认 `data/` 目录下存在 `.txt` 或 `.pdf` 文件。

### 4. 修改知识资料后回答没变化

删除 `md5.text`，必要时同时删除 `chroma_db/`，然后重新执行知识库构建命令。

### 5. 报告没有切换到报告提示词

检查日志：

```text
logs/agent_YYYYMMDD.log
```

确认模型是否调用了 `fill_context_for_report` 工具。

## 提交 GitHub 前的建议

建议提交：

- 源码目录：`agent/`、`rag/`、`model/`、`utils/`
- 配置目录：`config/`
- 提示词目录：`prompts/`
- 示例数据：`data/`
- 文档：`README.md`、`docs/`
- 依赖：`requirements.txt`
- 忽略规则：`.gitignore`

不建议提交：

- `.env` 或任何真实密钥
- `chroma_db/`
- `md5.text`
- `logs/`
- `__pycache__/`
- IDE 配置目录，如 `.idea/`

## License

如果你计划开源发布，建议补充许可证文件，例如 MIT License。

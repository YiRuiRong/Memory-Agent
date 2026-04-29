# Quickly Starter — 扫地机器人智能客服 Agent 项目

这是一份 **从零开始读懂本项目** 的快速入门指南。按顺序看完，你就能掌握项目干了什么、每一块代码的职责、以及一次用户提问背后发生的全过程。

---

## 1. 一句话理解项目

> 做一个「扫地机器人售后智能客服」Web 应用：用户提问 → Agent 自主思考 → 调用工具（RAG 知识库 / 查天气 / 查用户ID / 查使用记录...） → 整合回答。还能根据意图自动切换提示词来写"使用报告"。

技术栈：**LangChain Agent (ReAct) + Chroma 向量库 + 通义千问 qwen3-max + DashScope Embedding + Streamlit**。

---

## 2. 项目能做什么（3 个典型场景）

| 场景 | 用户输入示例 | Agent 内部动作 |
|------|---------------|----------------|
| 售后咨询 | "我家机器人老是撞墙怎么办？" | 调 `rag_summarize` 检索 `data/` 中的故障排除文档 → 总结回答 |
| 环境适配建议 | "我在杭州，适合开扫地机吗？" | 调 `get_user_location` + `get_weather` → 结合天气给建议 |
| 生成使用报告 | "给我生成我的使用报告" | 调 `get_user_id` → `get_current_month` → `fill_context_for_report` 触发提示词切换 → `fetch_external_data` → 写 Markdown 报告 |

---

## 3. 启动前的准备

### 3.1 环境变量
本项目使用阿里云 DashScope，需设置：
```bash
export DASHSCOPE_API_KEY="你的key"    # Windows 用 set 或在系统变量中配置
```

### 3.2 依赖（核心包）
```text
streamlit
langchain
langchain-chroma
langchain-community
langchain-core
langchain-text-splitters
pyyaml
pypdf            # PyPDFLoader 依赖
```

### 3.3 第一次启动的三步
```bash
# 1) 先把 data/ 下的文档灌入向量库（只做一次，增量靠 md5.text 去重）
python -m rag.vector_store

# 2) 启动前端
streamlit run app.py

# 3) 浏览器打开 http://localhost:8501，开始提问
```

> ⚠️ 所有 `python` 命令都要用 `-m` 模块方式启动，且在 **项目根目录** 执行。因为代码里大量使用 `from utils.xxx import`，直接 `python rag/vector_store.py` 会报导入错误。

---

## 4. 项目目录速览

```
├── app.py                      # Streamlit 入口（极简UI+流式输出）
├── agent/
│   ├── react_agent.py          # ReactAgent类，用 create_agent 组装 Agent
│   └── tools/
│       ├── agent_tools.py      # 7个工具：RAG/天气/用户ID/月份/外部数据/信号工具
│       └── middleware.py       # 3个中间件：工具监控/模型前日志/动态提示词切换
├── rag/
│   ├── vector_store.py         # Chroma封装 + 文档切片入库（MD5去重）
│   └── rag_service.py          # RAG总结链，被agent的rag_summarize工具调用
├── model/factory.py            # 工厂模式创建 chat_model / embed_model 单例
├── prompts/
│   ├── main_prompt.txt         # 默认系统提示词（客服人设）
│   ├── report_prompt.txt       # 报告生成场景切换用的提示词
│   └── rag_summarize.txt       # RAG总结链的用户提示词模板
├── config/
│   ├── agent.yml               # 外部CSV路径
│   ├── chroma.yml              # 向量库配置（分片大小/重叠/k/md5账本...）
│   ├── prompts.yml             # 3个提示词文件的路径
│   └── rag.yml                 # 模型名
├── utils/
│   ├── path_tool.py            # 相对路径 → 绝对路径
│   ├── config_handler.py       # 加载4个yaml为全局dict
│   ├── prompt_loader.py        # 读取3个提示词文件
│   ├── file_handler.py         # MD5/txt/pdf加载
│   └── logger_handler.py       # 全局logger，日志按日期切分到 logs/
├── data/                       # 知识库原始文档（扫地机器人100问.pdf 等）
│   └── external/records.csv    # 模拟"外部业务系统"的用户使用记录
├── chroma_db/                  # Chroma持久化目录（运行后生成）
├── logs/                       # 日志（运行后生成）
└── md5.text                    # 已入库文件的md5账本（用于去重）
```

---

## 5. 核心概念：ReAct Agent 是怎么跑起来的

### 5.1 Agent 的三块拼装件（看 `agent/react_agent.py`）

```python
self.agent = create_agent(
    model=chat_model,                  # 大模型
    system_prompt=load_system_prompts(),# 系统提示词
    tools=[...7个工具...],              # 工具清单
    middleware=[...3个中间件...],        # 中间件
)
```

LangChain 的 `create_agent` 会按 ReAct 思路：
**「思考 → 选工具 → 执行 → 看结果 → 再思考 → ... → 回答」** 自动循环，直到模型决定给终答。

### 5.2 工具清单（`agent/tools/agent_tools.py`）

| 工具 | 入参 | 干嘛的 | 是否mock |
|------|------|--------|----------|
| `rag_summarize` | query | 从 Chroma 里检索文档 + 模型总结 | 真实 |
| `get_weather` | city | 返回固定的"晴天26度"字符串 | Mock |
| `get_user_location` | 无 | 随机返回深圳/合肥/杭州 | Mock |
| `get_user_id` | 无 | 随机返回 1001-1010 | Mock |
| `get_current_month` | 无 | 随机返回 2025-01~12 | Mock |
| `fetch_external_data` | user_id, month | 从 `records.csv` 查该用户该月记录 | 真实（模拟外部系统） |
| `fill_context_for_report` | 无 | **信号工具**，不做业务，只为触发中间件 | — |

> 注：mock 工具返回随机值，是为了模拟业务系统；生产环境你只需替换实现即可。

### 5.3 中间件（`agent/tools/middleware.py`）

三个中间件分别挂在不同生命周期：

- **`@wrap_tool_call` monitor_tool**：每次工具执行前后打日志；**并顺便拦截 `fill_context_for_report` 把 `context["report"] = True`**
- **`@before_model` log_before_model**：每次调大模型前打日志
- **`@dynamic_prompt` report_prompt_switch**：**每次生成提示词时动态选择**——如果 `context["report"]==True` 用报告提示词，否则用默认客服提示词

---

## 6. 重点机制：动态提示词切换（整个项目的点睛之笔）

这是本项目最"非直观"的部分，贯穿 **4 个文件**，务必理解：

```
用户："给我生成我的使用报告"
     │
     ▼
main_prompt.txt（提示词要求报告场景必须先调 fill_context_for_report）
     │
     ▼
模型输出：调用 fill_context_for_report 工具
     │
     ▼
middleware.py::monitor_tool 拦截：
    if tool_name == "fill_context_for_report":
        runtime.context["report"] = True   ←← 翻转开关
     │
     ▼
下一轮模型调用前：
middleware.py::report_prompt_switch (@dynamic_prompt) 触发：
    if context["report"]: return load_report_prompts()   ←← 换成报告提示词
    else: return load_system_prompts()
     │
     ▼
模型在"报告提示词"人设下继续调用 get_user_id / get_current_month / fetch_external_data
     │
     ▼
按 report_prompt.txt 要求输出 Markdown 报告
```

**关键点**：
- `fill_context_for_report` 本身返回固定字符串"已调用"，没业务逻辑——它是一个 **信号工具**。
- 修改这个工具的名字，必须同步改 `monitor_tool` 里的字符串判断、`main_prompt.txt` 里的调用指引。
- `context={"report": False}` 的初始值在 `react_agent.py::execute_stream` 传入。

---

## 7. RAG 管线：文档怎么变成回答

### 7.1 灌库阶段（`rag/vector_store.py`）
```
data/*.txt, *.pdf
    │
    │ listdir_with_allowed_type 扫描
    ▼
每个文件算 MD5 → 查 md5.text → 已灌过就跳过
    │
    ▼
TextLoader / PyPDFLoader 读成 Document
    │
    ▼
RecursiveCharacterTextSplitter 切片（chunk_size=200, overlap=20, 中文优先分隔符）
    │
    ▼
Chroma.add_documents（DashScope embedding）
    │
    ▼
成功后把 md5 追加进 md5.text
```

> **去重的关键是 `md5.text`**：每行一个十六进制 MD5。如果你想"重新灌一次"，删掉这个文件即可（注意：Chroma 持久化目录里旧向量还在，严格重建要一并删 `chroma_db/`）。

### 7.2 查询阶段（`rag/rag_service.py`）
```
rag_summarize(query)
    │
    ▼
retriever.invoke(query) → 取 top-k（k=3）文档
    │
    ▼
拼成 "【参考资料1】..." 的 context 字符串
    │
    ▼
PromptTemplate(rag_summarize.txt) | print_prompt | chat_model | StrOutputParser
    │
    ▼
返回纯文本总结
```

---

## 8. 配置体系：改什么去哪里改

| 想改的东西 | 改哪个文件 |
|-----------|-----------|
| 换对话模型 / 换 embedding 模型 | `config/rag.yml` |
| 切片大小、检索 topK、向量库目录、允许的文件后缀 | `config/chroma.yml` |
| 提示词文件路径 | `config/prompts.yml`（默认已指向 `prompts/*.txt`） |
| 外部业务数据 CSV 路径 | `config/agent.yml` |
| 客服人设/工具使用规则 | `prompts/main_prompt.txt` |
| 报告人设/输出格式 | `prompts/report_prompt.txt` |
| RAG 总结的回答约束 | `prompts/rag_summarize.txt` |

**所有路径都是相对项目根目录**，代码中统一用 `utils.path_tool.get_abs_path(相对路径)` 解析。不要把 `utils/` 挪位置，否则根目录推断会错（它是从 `__file__` 向上 2 层拿的）。

---

## 9. 一次完整调用的时序（把前面都串起来）

以"我家机器人拖地不干净"为例：

```
1. app.py 收到用户输入
2. ReactAgent.execute_stream(prompt) 启动
3. agent.stream(..., context={"report": False})
4. @dynamic_prompt 选择 main_prompt.txt 作为系统提示词
5. @before_model 日志：即将调用模型，1条消息
6. chat_model 思考 → 决定调 rag_summarize("拖地不干净")
7. @wrap_tool_call 日志：执行工具 rag_summarize
8.   rag_service 内部：retriever 取 3 条文档 → chain 调模型总结
9. 工具返回字符串
10. @before_model 日志：即将调用模型，3条消息（user / ai / tool）
11. chat_model 判断信息够了 → 生成最终回答
12. execute_stream 按 chunk yield 给 app.py
13. st.write_stream 逐字显示
```

如果是报告场景，在第 6 步模型会先调 `fill_context_for_report` → 中间件翻转 context → 再次回到第 4 步但这次拿到的是 `report_prompt.txt` → 整个人设切换为报告写手。

---

## 10. 建议的学习路径（按顺序看）

1. **`app.py`** —— 看入口最直观，30 秒明白输入输出
2. **`model/factory.py`** + **`utils/config_handler.py`** —— 理解模型和配置怎么进来的
3. **`utils/path_tool.py`** + **`utils/prompt_loader.py`** —— 理解资源路径和提示词加载
4. **`agent/react_agent.py`** —— Agent 的组装
5. **`agent/tools/agent_tools.py`** —— 工具都长啥样
6. **`agent/tools/middleware.py`** ⭐ —— 最难也最关键的部分，反复看
7. **`prompts/main_prompt.txt`** + **`prompts/report_prompt.txt`** —— 结合第 6 步看提示词切换
8. **`rag/vector_store.py`** + **`rag/rag_service.py`** —— RAG 的两段（灌库 + 查询）
9. **`utils/file_handler.py`** + **`utils/logger_handler.py`** —— 工具类扫一遍就行

---

## 11. 常见坑

- **启动报 `ModuleNotFoundError: utils`**：你没在项目根目录运行；或者用了 `python xxx.py` 而不是 `python -m xxx.yyy`。
- **启动报 `DASHSCOPE_API_KEY` 缺失**：环境变量没设。
- **知识库加载只有一次，改了文档没生效**：删 `md5.text`（或只删对应那行）重跑 `python -m rag.vector_store`。
- **Chroma 重建**：删掉根目录的 `chroma_db/` 整个目录 + 删 `md5.text`，再灌库。
- **项目里有 3 个 `chroma_db/` 目录**（根目录、`agent/`、`rag/`）：只有 **根目录下的** 是真正在用的（由 `chroma.yml` 配 `persist_directory: chroma_db`，经 `get_abs_path` 转绝对路径后落在项目根）。其它是以前在子目录运行脚本残留的，可忽略。
- **报告没切换人设**：检查模型是否真的调用了 `fill_context_for_report`，看 `logs/agent_YYYYMMDD.log` 里 `[tool monitor]` 有没有对应记录。

---

## 12. 动手改造练习（学完后试试）

1. 加一个新工具 `get_device_battery(device_id)` 返回 mock 电量，让 Agent 在用户问"机器人电量"时调用。
2. 把 `rag.yml` 里的 `chat_model_name` 换成 `qwen-plus`，看响应速度和质量区别。
3. 把切片的 `chunk_size` 从 200 改成 500，重建向量库，对比 RAG 回答的差异。
4. 新增一个"售后投诉"提示词文件，仿照 `fill_context_for_report` 加一个信号工具 `fill_context_for_complaint`，让 Agent 识别投诉意图后切到该提示词。

做完这 4 个你就完全掌握本项目了。

---

_最后更新：2026-04-24_

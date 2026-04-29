"""
当前 LangChain agent 使用的 GenericAgent 风格分层文本记忆。

持久层位于项目根目录的 memory/ 下：
- L1：global_mem_insight.txt，短小、常驻注入 prompt 的索引。
- L2：global_mem.txt，稳定的跨任务事实库。
- L3：sop_library.md，可复用流程和坑点。
- L4：L4_raw_sessions/，压缩后的历史会话归档。
"""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from utils.logger_handler import logger
from utils.path_tool import get_abs_path


MEMORY_ROOT = Path(get_abs_path("memory"))
L1_PATH = MEMORY_ROOT / "global_mem_insight.txt"
L2_PATH = MEMORY_ROOT / "global_mem.txt"
L3_PATH = MEMORY_ROOT / "sop_library.md"
L0_PATH = MEMORY_ROOT / "memory_management_sop.md"
L4_DIR = MEMORY_ROOT / "L4_raw_sessions"
FILE_ACCESS_STATS_PATH = MEMORY_ROOT / "file_access_stats.json"
CURRENT_SESSION_LOG_PATH = L4_DIR / "current_session.log"
ALL_HISTORIES_PATH = L4_DIR / "all_histories.txt"
CURRENT_SESSION_DONE_PATH = L4_DIR / "current_session.log.done"
ARCHIVE_LOCK_PATH = L4_DIR / "archive.lock"
CURRENT_SESSION_MAX_BYTES = 512 * 1024
MEMORY_ANCHOR_MESSAGE_ID = "memory-anchor-runtime"

EXPLICIT_MEMORY_TRIGGERS = (
    "记住",
    "保存到记忆",
    "更新 memory",
    "更新memory",
    "沉淀",
    "复盘",
    "下次按这个流程",
    "以后都这样",
    "以后按这个",
    "写成 SOP",
    "写成SOP",
    "加到 L1",
    "加到L1",
    "加到 L2",
    "加到L2",
    "加到 L3",
    "加到L3",
    "这个配置固定",
    "这个路径以后用",
    "别再犯这个错",
)

PREFERENCE_TRIGGERS = (
    "我偏好",
    "我喜欢",
    "我希望",
    "以后默认",
    "以后都",
    "团队约定",
    "下次优先",
    "不要再",
)

SOP_TRIGGERS = (
    "SOP",
    "流程",
    "步骤",
    "攻略",
    "排查",
    "保养",
    "脚本",
    "方法",
    "复盘",
    "下次按",
)

STABLE_FACT_TRIGGERS = (
    "路径",
    "配置",
    "固定",
    "项目",
    "约束",
    "目录",
    ".py",
    ".yml",
    ".yaml",
    ".json",
    ".md",
    ".txt",
)

VOLATILE_TRIGGERS = (
    "当前天气",
    "实时天气",
    "今天",
    "现在",
    "本次",
    "临时",
    "随机",
    "当前月份",
    "当前用户",
    "气温",
    "湿度",
)

SECRET_PATTERNS = (
    r"api[_-]?key\s*[:=]",
    r"secret\s*[:=]",
    r"token\s*[:=]",
    r"password\s*[:=]",
    r"AKIA[0-9A-Z]{16}",
    r"sk-[A-Za-z0-9_-]{12,}",
)

VOLATILE_TOOL_NAMES = {
    "get_weather",
    "get_user_location",
    "get_user_id",
    "get_current_month",
    "fetch_external_data",
}

REFERENTIAL_MEMORY_WORDS = (
    "这个",
    "这份",
    "这条",
    "这段",
    "这个策略",
    "这个保养策略",
    "这个保养攻略",
    "这个流程",
    "这个方法",
    "上面",
    "上述",
    "前面",
    "刚才",
    "上一条",
    "这次",
)


DEFAULT_L1 = """# [全局记忆索引 - L1]

L1 是短小的全局记忆索引，会常驻注入 system prompt。它只提供持久记忆的地图，不复制完整内容。

## 高价值指针
- L2 稳定事实：只有在需要项目事实、路径或环境约束时读取 `memory/global_mem.txt`。
- L3 可复用 SOP：修改 agent prompt、中间件或工具链路前，先读取 `memory/sop_library.md`。
- L4 会话归档：只有在需要历史会话轨迹时读取 `memory/L4_raw_sessions/`。
- 运行时锚点：每轮模型调用都会收到最近的 `[USER]` / `[Agent]` 摘要历史和 `<key_info>`。
- 自动沉淀：用户明确说“记住/沉淀/写成 SOP/以后按此流程”时，先收集候选并校验证据，再写入 L1/L2/L3。
- 回复边界：长期记忆写入在模型回复结束后由程序 hook 执行；模型不得声称“正在写入/已经写入”某个具体层级或文件，只能说明“会提交给记忆系统按规则评估和沉淀”。

## 项目指针
- 主应用入口：`app.py`。
- LangChain agent 运行器：`agent/react_agent.py`。
- system prompt 加载器：`utils/prompt_loader.py`。
- agent 中间件与动态提示词切换：`agent/tools/middleware.py`。
"""


DEFAULT_L2 = """# [全局事实库 - L2]

L2 保存已验证、稳定、跨任务有效的事实。不要写入临时计划、易变状态、secret 或未验证猜测。

## 已验证项目事实
- 当前项目是一个 Streamlit + LangChain agent，用于扫地/扫拖机器人智能客服。
- UI 入口是 `app.py`，并在 `st.session_state` 中保存一个 `ReactAgent` 实例。
- 核心 agent 在 `agent/react_agent.py` 中通过 `langchain.agents.create_agent` 构建。
- 基础提示词位于 `prompts/`，提示词路径由 `config/prompts.yml` 配置。
- 运行日志写入 `logs/`。
"""


DEFAULT_L3 = """# [专项 SOP 库 - L3]

L3 保存可复用流程、集成说明和坑点。只写入已验证的工作流。

## Agent Prompt / 记忆集成 SOP
- 持久记忆 helper 放在 agent 运行器附近；当前项目使用 `agent/memory_system.py`。
- 默认只把 L1 注入 system prompt；L2/L3/L4 按需读取。
- 在模型调用前通过 middleware 注入短运行时锚点，避免把长记忆追加到每个用户 prompt。
- 保留报告提示词切换机制：报告模式仍使用报告 prompt，然后追加全局记忆。
- 自动沉淀链路：`record_user_turn` 检测显式记忆意图，`record_tool_result` 收集工具证据，`turn_end_callback` 触发候选分类和 `apply_memory_update` 写入。
"""


DEFAULT_L0 = """# 记忆管理 SOP - L0

写入 L1/L2/L3 前必须遵守这些规则。

## 分层规则
- 无行动，不记忆：只写入来自成功工具调用、测试、文件读取、运行日志或用户明确事实的信息。
- L1 是地图，不是教程。保持短小，只放高价值指针。
- L2 保存稳定全局事实：路径、配置、环境约束和跨任务有效事实。
- L3 保存可复用 SOP、脚本、坑点和可重复执行的流程。
- L4 保存压缩后的历史会话，默认不注入 prompt；归档后必须截断、轮转或删除已处理源日志，避免 active log 无限增长。
- 自动沉淀必须先执行：候选提取 -> 证据校验 -> 分层分类 -> 读取现有 memory -> 最小 patch -> 校验写入。

## 禁止写入
- secret、凭证、API key、个人敏感数据或未脱敏的私有记录。
- 当前 PID、临时浏览器 tab、一次性工作的时间戳、临时计划或未验证猜测。
- 大段原始工具输出，除非已经压缩且对 L4 有长期价值。

## Patch 纪律
- 优先使用最小追加或 patch 更新。
- 只有确认旧事实已经过期时，才删除或改写持久记忆。
- 不确定时，在 L1 留短指针，把详细流程放入 L3。
"""


DEFAULT_FILE_ACCESS_STATS = """{
  "description": "memory/SOP 文件的可选访问统计。",
  "files": {}
}
"""


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _write_if_missing(path: Path, content: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _truncate(text: str, limit: int = 160) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _strip_code_blocks(text: str) -> str:
    return re.sub(r"```.*?```", "", text or "", flags=re.S)


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword.lower() in (text or "").lower() for keyword in keywords)


def _contains_secret(text: str) -> bool:
    return any(re.search(pattern, text or "", flags=re.I) for pattern in SECRET_PATTERNS)


def _looks_volatile(text: str) -> bool:
    return _contains_any(text, VOLATILE_TRIGGERS)


def _looks_like_sop(text: str) -> bool:
    return _contains_any(text, SOP_TRIGGERS)


def _looks_like_stable_fact(text: str) -> bool:
    return _contains_any(text, STABLE_FACT_TRIGGERS)


def _looks_like_preference(text: str) -> bool:
    return _contains_any(text, PREFERENCE_TRIGGERS)


def _has_explicit_memory_intent(text: str) -> bool:
    return _contains_any(text, EXPLICIT_MEMORY_TRIGGERS)


def _is_referential_memory_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if compact in {re.sub(r"\s+", "", item) for item in REFERENTIAL_MEMORY_WORDS}:
        return True
    return any(word in compact for word in ("这个", "这份", "这条", "这段", "上面", "上述", "前面", "刚才"))


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _short_title(text: str) -> str:
    compact = _truncate(text, 42)
    compact = re.sub(r"^[#\-*\s]+", "", compact).strip()
    return compact or "自动沉淀条目"


def strip_summary_blocks(text: str) -> str:
    """展示给用户前，移除可见的 summary 块。"""
    return re.sub(r"<summary>.*?</summary>", "", text or "", flags=re.S | re.I).strip()


def _extract_summary(text: str) -> str:
    clean = _strip_code_blocks(text)
    match = re.search(r"<summary>(.*?)</summary>", clean, flags=re.S | re.I)
    if match:
        return _truncate(match.group(1), 100)
    return _truncate(strip_summary_blocks(clean), 100)


def _message_content(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _is_memory_anchor(message: BaseMessage) -> bool:
    if getattr(message, "id", None) == MEMORY_ANCHOR_MESSAGE_ID:
        return True
    content = _message_content(message)
    return "<runtime_memory>" in content and "<key_info>" in content


def _append_unique(path: Path, content: str) -> bool:
    content = (content or "").strip()
    if not content:
        return False

    existing = _read_text(path)
    if content in existing:
        return False

    separator = "" if existing.endswith("\n") or not existing else "\n"
    path.write_text(f"{existing}{separator}{content}\n", encoding="utf-8")
    return True


def ensure_memory_files() -> None:
    """创建持久记忆目录以及基础 L0/L1/L2/L3/L4 文件。"""
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    L4_DIR.mkdir(parents=True, exist_ok=True)

    _write_if_missing(L1_PATH, DEFAULT_L1)
    _write_if_missing(L2_PATH, DEFAULT_L2)
    _write_if_missing(L3_PATH, DEFAULT_L3)
    _write_if_missing(L0_PATH, DEFAULT_L0)
    _write_if_missing(FILE_ACCESS_STATS_PATH, DEFAULT_FILE_ACCESS_STATS)
    _write_if_missing(CURRENT_SESSION_LOG_PATH, "")
    _write_if_missing(ALL_HISTORIES_PATH, "")


def get_global_memory() -> str:
    """返回用于注入 system prompt 的 L1 记忆索引和固定分层说明。"""
    ensure_memory_files()
    insight = _read_text(L1_PATH).strip()
    return f"""

<global_memory>
记忆分层：
- L1 `memory/global_mem_insight.txt`：短小、常驻注入的索引。
- L2 `memory/global_mem.txt`：已验证的持久事实，只在需要时读取。
- L3 `memory/sop_library.md`：可复用 SOP 和集成说明，相关改动前先读取。
- L4 `memory/L4_raw_sessions/`：压缩后的会话归档，默认不注入。

规则：
- 不要写入 secret、临时状态或未验证猜测。
- L1 只作为地图使用，详细事实和 SOP 放在 L2/L3。
- 长期记忆写入由程序后置 hook 执行。模型回复用户时不能承诺已写入 L1/L2/L3/L4，也不能臆测目标层级；只能说会提交给记忆系统按规则评估、分类和沉淀。

{insight}
</global_memory>
""".strip()


def get_system_prompt(base_prompt: str) -> str:
    """在基础 system prompt 后追加全局记忆。"""
    return f"{base_prompt.rstrip()}\n\n{get_global_memory()}"


@dataclass
class MemoryCandidate:
    """等待校验、分类和沉淀的候选记忆。"""

    content: str
    source: str
    evidence: str
    scope: str = "session"
    stability: str = "unknown"
    target_layer: str = "reject"
    reason: str = ""
    intent: str = "implicit"
    trigger: str = ""
    tool_name: str = ""
    verified: bool = False


@dataclass
class AgentMemoryRuntime:
    """单个 agent 运行时的进程内工作记忆。"""

    history_info: list[str] = field(default_factory=list)
    key_info: str = ""
    related_sop: str = ""
    turn_count: int = 0
    max_history_items: int = 40
    max_session_messages: int = 24
    pending_candidates: list[MemoryCandidate] = field(default_factory=list)
    tool_results: list[dict[str, str]] = field(default_factory=list)
    last_user_message: str = ""
    last_summary: str = ""
    last_agent_response: str = ""
    explicit_memory_intent: bool = False

    def record_user_turn(self, query: str) -> None:
        query = _truncate(query, 160)
        if not query:
            return
        self.last_user_message = query
        if _has_explicit_memory_intent(query):
            self.explicit_memory_intent = True
        self.history_info.append(f"[USER] {query}")
        self.history_info = self.history_info[-self.max_history_items :]
        self._append_current_session_line(f"[USER] {query}")
        self.collect_memory_candidates(user_message=query)

    def update_working_checkpoint(self, key_info: str, related_sop: str = "") -> str:
        self.key_info = (key_info or "").strip()
        self.related_sop = (related_sop or "").strip()
        return self.get_anchor_prompt()

    def record_tool_result(
        self,
        tool_name: str,
        tool_args: Any,
        result: Any,
        success: bool = True,
    ) -> None:
        result_text = _truncate(getattr(result, "content", str(result)), 600)
        args_text = _truncate(str(tool_args), 240)
        tool_record = {
            "tool_name": tool_name,
            "args": args_text,
            "result": result_text,
            "success": str(success),
        }
        self.tool_results.append(tool_record)
        self.tool_results = self.tool_results[-12:]
        self.collect_memory_candidates(tool_results=[tool_record])

    def collect_memory_candidates(
        self,
        user_message: str = "",
        summary: str = "",
        tool_results: Sequence[dict[str, str]] | None = None,
    ) -> list[MemoryCandidate]:
        """从用户话术、summary 和工具结果中收集候选记忆。"""
        new_candidates: list[MemoryCandidate] = []
        user_message = user_message or self.last_user_message
        summary = summary or self.last_summary

        if user_message and _has_explicit_memory_intent(user_message):
            content = self._extract_explicit_memory_content(user_message, summary)
            evidence = f"用户明确要求长期记忆：{_truncate(user_message, 120)}"
            if summary:
                evidence += f"；本轮摘要：{_truncate(summary, 120)}"
            verified = (
                not _contains_secret(content)
                and (
                    _looks_like_preference(user_message)
                    or _looks_like_sop(content)
                    or _looks_like_stable_fact(content)
                    or _looks_like_stable_fact(user_message)
                )
            )
            new_candidates.append(
                MemoryCandidate(
                    content=content,
                    source="user",
                    evidence=evidence,
                    scope="global",
                    stability="durable",
                    intent="explicit",
                    trigger="explicit_memory_intent",
                    verified=verified,
                )
            )

        if user_message and _looks_like_preference(user_message):
            new_candidates.append(
                MemoryCandidate(
                    content=f"用户偏好或团队约定：{_truncate(user_message, 180)}",
                    source="user",
                    evidence=f"用户明确表达：{_truncate(user_message, 120)}",
                    scope="global",
                    stability="durable",
                    intent="implicit",
                    trigger="user_preference",
                    verified=True,
                )
            )

        for tool_result in tool_results or []:
            tool_name = tool_result.get("tool_name", "")
            result = tool_result.get("result", "")
            success = tool_result.get("success") == "True"
            if not success or not result:
                continue
            evidence = (
                f"工具 `{tool_name}` 调用成功；参数：{tool_result.get('args', '')}；"
                f"结果摘要：{_truncate(result, 180)}"
            )
            new_candidates.append(
                MemoryCandidate(
                    content=_truncate(result, 240),
                    source="tool_result",
                    evidence=evidence,
                    scope="task",
                    stability="durable" if tool_name not in VOLATILE_TOOL_NAMES else "volatile",
                    intent="implicit",
                    trigger="tool_result",
                    tool_name=tool_name,
                    verified=tool_name not in VOLATILE_TOOL_NAMES,
                )
            )

        if summary and self.turn_count + 1 >= 15:
            new_candidates.append(
                MemoryCandidate(
                    content=f"长任务完成摘要：{_truncate(summary, 180)}",
                    source="summary",
                    evidence=f"任务已达到 {self.turn_count + 1} 轮；摘要：{_truncate(summary, 120)}",
                    scope="task",
                    stability="unknown",
                    intent="implicit",
                    trigger="long_task_summary",
                    verified=False,
                )
            )

        for candidate in new_candidates:
            self._add_candidate(candidate)
        return new_candidates

    def should_distill_memory(self) -> bool:
        """判断当前候选队列是否应该进入长期沉淀。"""
        if not self.pending_candidates:
            return False
        if self.explicit_memory_intent:
            return True
        if self.turn_count + 1 >= 15 and any(c.source == "summary" for c in self.pending_candidates):
            return True
        return any(
            c.verified
            and c.source == "tool_result"
            and c.tool_name not in VOLATILE_TOOL_NAMES
            and (_looks_like_stable_fact(c.content) or _looks_like_sop(c.content))
            for c in self.pending_candidates
        )

    def classify_memory_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        """把候选记忆分类到 L1/L2/L3 或拒绝。"""
        text = f"{candidate.content}\n{candidate.evidence}"

        if _contains_secret(text):
            candidate.target_layer = "reject"
            candidate.reason = "包含疑似 secret、token、密码或密钥"
            return candidate

        if candidate.tool_name in VOLATILE_TOOL_NAMES:
            candidate.target_layer = "reject"
            candidate.reason = f"工具 `{candidate.tool_name}` 的结果属于易变或个人会话数据"
            return candidate

        if candidate.source == "manual" and candidate.target_layer in {"L1", "L2", "L3"}:
            candidate.reason = candidate.reason or "开发者手动指定目标层"
            return candidate

        if _looks_volatile(candidate.content) and not _looks_like_sop(candidate.content):
            candidate.target_layer = "reject"
            candidate.reason = "候选内容包含易变状态，不适合写入长期记忆"
            return candidate

        if not candidate.verified:
            candidate.target_layer = "reject"
            candidate.reason = "缺少工具结果、文件读取、运行日志或用户明确事实作为证据"
            return candidate

        if "加到 L1" in text or "加到L1" in text or ("L1" in candidate.content and "索引" in text):
            candidate.target_layer = "L1"
            candidate.reason = "用户明确要求写入 L1 或内容是全局索引指针"
        elif "加到 L3" in text or "加到L3" in text or _looks_like_sop(text):
            candidate.target_layer = "L3"
            candidate.reason = "候选是可复用流程、SOP、攻略、脚本或排查方法"
        elif "加到 L2" in text or "加到L2" in text or _looks_like_stable_fact(text):
            candidate.target_layer = "L2"
            candidate.reason = "候选是稳定事实、配置、路径、项目约束或用户明确偏好"
        elif candidate.source == "user" and candidate.intent == "explicit":
            candidate.target_layer = "L2"
            candidate.reason = "用户明确要求长期记忆，按用户事实/偏好写入 L2"
        else:
            candidate.target_layer = "reject"
            candidate.reason = "未达到 L1/L2/L3 的长期复用价值"

        return candidate

    def apply_memory_update(self, candidate: MemoryCandidate) -> dict[str, object]:
        """读取 L0 后最小 patch L1/L2/L3，并返回写入结果。"""
        ensure_memory_files()
        _read_text(L0_PATH)
        candidate = self.classify_memory_candidate(candidate)
        if candidate.target_layer == "reject":
            return {
                "target_layer": "reject",
                "written": False,
                "reason": candidate.reason,
                "content": candidate.content,
            }

        entry = self._format_candidate_entry(candidate)
        target_path = {
            "L1": L1_PATH,
            "L2": L2_PATH,
            "L3": L3_PATH,
        }[candidate.target_layer]
        written = _append_unique(target_path, entry)

        l1_pointer_written = False
        if written and candidate.target_layer in {"L2", "L3"}:
            pointer = self._format_l1_pointer(candidate)
            l1_pointer_written = _append_unique(L1_PATH, pointer)

        return {
            "target_layer": candidate.target_layer,
            "written": written,
            "l1_pointer_written": l1_pointer_written,
            "reason": candidate.reason,
            "content": candidate.content,
            "path": str(target_path),
        }

    def _extract_explicit_memory_content(self, user_message: str, summary: str = "") -> str:
        normalized = user_message.strip()
        for trigger in EXPLICIT_MEMORY_TRIGGERS:
            if trigger in normalized:
                tail = normalized.split(trigger, 1)[-1].strip(" ：:，,。")
                if tail and not _is_referential_memory_text(tail):
                    return _truncate(tail, 240)
        recent_agent = next(
            (item.replace("[Agent]", "", 1).strip() for item in reversed(self.history_info) if item.startswith("[Agent]")),
            "",
        )
        if self.last_agent_response:
            return _truncate(self.last_agent_response, 800)
        if recent_agent:
            return _truncate(recent_agent, 500)
        if summary:
            return _truncate(summary, 500)
        return _truncate(user_message, 240)

    def _add_candidate(self, candidate: MemoryCandidate) -> None:
        key = _dedupe_key(f"{candidate.content}|{candidate.source}|{candidate.trigger}")
        existing_keys = {
            _dedupe_key(f"{item.content}|{item.source}|{item.trigger}")
            for item in self.pending_candidates
        }
        if key not in existing_keys:
            self.pending_candidates.append(candidate)

    def _format_candidate_entry(self, candidate: MemoryCandidate) -> str:
        if candidate.target_layer == "L1":
            return (
                f"- {_short_title(candidate.content)} -> {candidate.content}"
                f"（来源：{candidate.source}；证据：{_truncate(candidate.evidence, 120)}；原因：{candidate.reason}）"
            )

        if candidate.target_layer == "L2":
            return f"""

## 自动沉淀事实
- 内容：{candidate.content}
  - 证据：{_truncate(candidate.evidence, 200)}
  - 来源：{candidate.source}
  - 分层原因：{candidate.reason}
""".strip()

        return f"""

## 自动沉淀 SOP：{_short_title(candidate.content)}
- 内容：{candidate.content}
- 证据：{_truncate(candidate.evidence, 220)}
- 来源：{candidate.source}
- 分层原因：{candidate.reason}
""".strip()

    def _format_l1_pointer(self, candidate: MemoryCandidate) -> str:
        target = "memory/global_mem.txt" if candidate.target_layer == "L2" else "memory/sop_library.md"
        label = _short_title(candidate.content)
        return f"- {label} -> `{target}`（自动沉淀，证据：{_truncate(candidate.evidence, 80)}）"

    def get_anchor_prompt(self) -> str:
        ensure_memory_files()
        recent_history = "\n".join(self.history_info[-self.max_history_items :])
        if not recent_history:
            recent_history = "[空]"

        key_info = self.key_info or "[空]"
        sop_hint = self.related_sop or "[空]"

        return f"""<runtime_memory>
<history>
{recent_history}
</history>
当前轮次：{self.turn_count}
<key_info>
{key_info}
</key_info>
<related_sop>
{sop_hint}
</related_sop>
将这段运行时记忆作为上下文使用。除非用户明确询问记忆内部实现，否则不要暴露这些标签。
如果用户要求“记住/沉淀/写成 SOP”，只说明系统会在本轮回复后评估候选记忆；不要声称已经写入 L1/L2/L3/L4 或正在写入某个具体文件。
</runtime_memory>"""

    def turn_end_callback(self, response: str, user_message: str = "") -> str:
        visible_response = _truncate(strip_summary_blocks(response), 1200)
        summary = _extract_summary(response)
        if summary:
            self.last_summary = summary
            self.history_info.append(f"[Agent] {summary}")
            self.history_info = self.history_info[-self.max_history_items :]
            self._append_current_session_line(f"[Agent] {summary}")
        self.collect_memory_candidates(
            user_message=user_message or self.last_user_message,
            summary=summary,
            tool_results=self.tool_results,
        )
        if self.should_distill_memory():
            self.start_long_term_update()
        self.tool_results.clear()
        self.explicit_memory_intent = False
        if visible_response:
            self.last_agent_response = visible_response
        self.turn_count += 1
        return summary

    def trim_session_history(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        non_anchor_messages = [msg for msg in messages if not _is_memory_anchor(msg)]
        if len(non_anchor_messages) <= self.max_session_messages:
            return list(non_anchor_messages)

        return list(non_anchor_messages[-self.max_session_messages :])

    def build_anchor_state_update(self, messages: Sequence[BaseMessage]) -> dict[str, list[BaseMessage]]:
        trimmed_messages = self.trim_session_history(messages)
        anchor = HumanMessage(content=self.get_anchor_prompt(), id=MEMORY_ANCHOR_MESSAGE_ID)
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trimmed_messages,
                anchor,
            ]
        }

    def start_long_term_update(
        self,
        verified_facts: str = "",
        sop_notes: str = "",
        insight_notes: str = "",
        candidates: Sequence[MemoryCandidate] | None = None,
    ) -> dict[str, object]:
        """
        应用已验证的持久记忆更新。

        调用方只能传入已验证事实或流程。没有参数时只返回 L0，
        便于调用方判断哪些内容可以安全沉淀。
        """
        ensure_memory_files()
        management_sop = _read_text(L0_PATH)
        update_candidates: list[MemoryCandidate] = list(candidates or [])

        if insight_notes:
            update_candidates.append(
                MemoryCandidate(
                    content=insight_notes,
                    source="manual",
                    evidence="开发者显式调用 start_long_term_update(insight_notes=...)",
                    scope="global",
                    stability="durable",
                    target_layer="L1",
                    reason="手动指定写入 L1",
                    intent="explicit",
                    verified=True,
                )
            )
        if verified_facts:
            update_candidates.append(
                MemoryCandidate(
                    content=verified_facts,
                    source="manual",
                    evidence="开发者显式调用 start_long_term_update(verified_facts=...)",
                    scope="global",
                    stability="durable",
                    target_layer="L2",
                    reason="手动指定写入 L2",
                    intent="explicit",
                    verified=True,
                )
            )
        if sop_notes:
            update_candidates.append(
                MemoryCandidate(
                    content=sop_notes,
                    source="manual",
                    evidence="开发者显式调用 start_long_term_update(sop_notes=...)",
                    scope="task",
                    stability="durable",
                    target_layer="L3",
                    reason="手动指定写入 L3",
                    intent="explicit",
                    verified=True,
                )
            )

        if not update_candidates:
            update_candidates = list(self.pending_candidates)

        writes = [self.apply_memory_update(candidate) for candidate in update_candidates]
        applied_keys = {
            _dedupe_key(str(item.get("content", "")))
            for item in writes
            if item.get("written") or item.get("target_layer") == "reject"
        }
        self.pending_candidates = [
            candidate
            for candidate in self.pending_candidates
            if _dedupe_key(candidate.content) not in applied_keys
        ]
        return {
            "memory_management_sop": management_sop,
            "writes": writes,
            "paths": {
                "L1": str(L1_PATH),
                "L2": str(L2_PATH),
                "L3": str(L3_PATH),
                "L4": str(L4_DIR),
            },
        }

    def archive_sessions(self, min_age_hours: int = 2) -> int:
        return archive_sessions(min_age_hours=min_age_hours)

    def finalize_session_log(self) -> Path:
        return finalize_session_log()

    def rotate_or_truncate_session_log(self) -> bool:
        return rotate_or_truncate_session_log()

    def _append_current_session_line(self, line: str) -> None:
        ensure_memory_files()
        with CURRENT_SESSION_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(line.strip() + "\n")


_DEFAULT_MEMORY = AgentMemoryRuntime()


def get_default_memory() -> AgentMemoryRuntime:
    ensure_memory_files()
    return _DEFAULT_MEMORY


def update_working_checkpoint(key_info: str, related_sop: str = "") -> str:
    return get_default_memory().update_working_checkpoint(key_info, related_sop)


def get_anchor_prompt() -> str:
    return get_default_memory().get_anchor_prompt()


def turn_end_callback(response: str, user_message: str = "") -> str:
    return get_default_memory().turn_end_callback(response, user_message=user_message)


def collect_memory_candidates(
    user_message: str = "",
    summary: str = "",
    tool_results: Sequence[dict[str, str]] | None = None,
) -> list[MemoryCandidate]:
    return get_default_memory().collect_memory_candidates(
        user_message=user_message,
        summary=summary,
        tool_results=tool_results,
    )


def should_distill_memory() -> bool:
    return get_default_memory().should_distill_memory()


def classify_memory_candidate(candidate: MemoryCandidate) -> MemoryCandidate:
    return get_default_memory().classify_memory_candidate(candidate)


def apply_memory_update(candidate: MemoryCandidate) -> dict[str, object]:
    return get_default_memory().apply_memory_update(candidate)


def start_long_term_update(
    verified_facts: str = "",
    sop_notes: str = "",
    insight_notes: str = "",
    candidates: Sequence[MemoryCandidate] | None = None,
) -> dict[str, object]:
    return get_default_memory().start_long_term_update(
        verified_facts=verified_facts,
        sop_notes=sop_notes,
        insight_notes=insight_notes,
        candidates=candidates,
    )


def trim_session_history(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    return get_default_memory().trim_session_history(messages)


def build_anchor_state_update(
    messages: Sequence[BaseMessage],
    memory: AgentMemoryRuntime | None = None,
) -> dict[str, list[BaseMessage]]:
    runtime_memory = memory or get_default_memory()
    return runtime_memory.build_anchor_state_update(messages)


@contextmanager
def _archive_lock() -> Iterable[None]:
    """保护 L4 归档、去重和 active log 截断这组原子操作。"""

    L4_DIR.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    for _ in range(2):
        try:
            fd = os.open(str(ARCHIVE_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stale_before = datetime.now() - timedelta(minutes=10)
                lock_mtime = datetime.fromtimestamp(ARCHIVE_LOCK_PATH.stat().st_mtime)
                if lock_mtime < stale_before:
                    ARCHIVE_LOCK_PATH.unlink()
                    continue
            except FileNotFoundError:
                continue
            raise RuntimeError(f"L4 归档锁仍被占用：{ARCHIVE_LOCK_PATH}")

    if fd is None:
        raise RuntimeError(f"无法创建 L4 归档锁：{ARCHIVE_LOCK_PATH}")

    try:
        os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}".encode("utf-8"))
        yield
    finally:
        os.close(fd)
        try:
            ARCHIVE_LOCK_PATH.unlink()
        except FileNotFoundError:
            pass


def _extract_history_lines(text: str) -> Iterable[str]:
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("[USER]") or line.startswith("[Agent]"):
            yield line

    for block in re.findall(r"<history>(.*?)</history>", text or "", flags=re.S | re.I):
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith("[USER]") or line.startswith("[Agent]"):
                yield line


def finalize_session_log() -> Path:
    """显式标记当前会话日志已完成，允许下一次归档不等待 mtime 阈值。"""

    ensure_memory_files()
    CURRENT_SESSION_DONE_PATH.write_text(datetime.now().isoformat(), encoding="utf-8")
    return CURRENT_SESSION_DONE_PATH


def rotate_or_truncate_session_log(path: Path = CURRENT_SESSION_LOG_PATH) -> bool:
    """
    截断已安全归档的 active session log。

    只应在 `archive_sessions()` 判定日志已完成或超过 mtime 安全阈值后调用；
    最近仍在写入的 `current_session.log` 会被 `archive_sessions()` 跳过。
    """

    ensure_memory_files()
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return False

    path.write_text("", encoding="utf-8")
    try:
        CURRENT_SESSION_DONE_PATH.unlink()
    except FileNotFoundError:
        pass
    return True


def archive_sessions(min_age_hours: int = 2) -> int:
    """
    将会话轨迹压缩归档到 L4。

    扫描当前记忆会话日志和项目日志中的 `[USER]` / `[Agent]`
    摘要。默认跳过最近修改过且未超限的原始文件；当前会话日志在
    成功归档、确认已去重或超过大小阈值后会被截断，避免 active log 无限增长。
    """
    ensure_memory_files()
    cutoff = datetime.now() - timedelta(hours=min_age_hours)
    candidates = [CURRENT_SESSION_LOG_PATH, *Path(get_abs_path("logs")).glob("*.log")]

    appended = 0

    with _archive_lock():
        existing_archive = _read_text(ALL_HISTORIES_PATH)
        with ALL_HISTORIES_PATH.open("a", encoding="utf-8") as archive:
            for path in candidates:
                if not path.exists() or path == ALL_HISTORIES_PATH:
                    continue

                is_current_session = path == CURRENT_SESSION_LOG_PATH
                size = path.stat().st_size
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                finalized = is_current_session and CURRENT_SESSION_DONE_PATH.exists()
                oversized = is_current_session and size > CURRENT_SESSION_MAX_BYTES
                if modified > cutoff and not finalized and not oversized:
                    continue

                text = _read_text(path)
                history_lines = list(_extract_history_lines(text))
                lines = [line for line in history_lines if line not in existing_archive]
                if lines:
                    archive.write(f"\n## 归档来源：{path.name}\n")
                    for line in lines:
                        archive.write(line + "\n")
                        appended += 1
                    existing_archive += "\n".join(lines)

                if is_current_session and text:
                    rotate_or_truncate_session_log(path)

    if appended:
        logger.info(f"[memory archive] 已将 {appended} 条摘要归档到 L4")
    return appended

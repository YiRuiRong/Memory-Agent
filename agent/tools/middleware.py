from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from agent.memory_system import build_anchor_state_update, get_default_memory

@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        memory = request.runtime.context.get("memory") or get_default_memory()
        request.runtime.context["memory"] = memory
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        if tool_name == "update_working_checkpoint" and isinstance(tool_args, dict):
            memory.update_working_checkpoint(
                key_info=tool_args.get("key_info", ""),
                related_sop=tool_args.get("related_sop", ""),
            )
        else:
            memory.record_tool_result(
                tool_name=tool_name,
                tool_args=tool_args,
                result=result,
                success=True,
            )

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        memory = request.runtime.context.get("memory") or get_default_memory()
        request.runtime.context["memory"] = memory
        memory.record_tool_result(
            tool_name=request.tool_call["name"],
            tool_args=request.tool_call.get("args", {}),
            result=str(e),
            success=False,
        )
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e

@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志，并注入短期工作记忆锚点
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    latest_content = getattr(state["messages"][-1], "content", "")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {str(latest_content).strip()}")

    memory = runtime.context.get("memory") or get_default_memory()
    runtime.context["memory"] = memory
    return build_anchor_state_update(state["messages"], memory=memory)

@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()

from langchain.agents import create_agent
from agent.memory_system import AgentMemoryRuntime, ensure_memory_files, strip_summary_blocks
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report,
                                     update_working_checkpoint)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        ensure_memory_files()
        self.memory = AgentMemoryRuntime()
        self.memory.archive_sessions()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report,
                   update_working_checkpoint],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        self.memory.record_user_turn(query)
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        final_content = ""

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False, "memory": self.memory}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                final_content = str(latest_message.content).strip()
                visible_content = strip_summary_blocks(final_content)
                if visible_content:
                    yield visible_content + "\n"

        if final_content:
            self.memory.turn_end_callback(final_content, user_message=query)

    def start_long_term_update(self, verified_facts: str = "", sop_notes: str = "", insight_notes: str = ""):
        return self.memory.start_long_term_update(
            verified_facts=verified_facts,
            sop_notes=sop_notes,
            insight_notes=insight_notes,
        )

    def archive_sessions(self, min_age_hours: int = 2) -> int:
        return self.memory.archive_sessions(min_age_hours=min_age_hours)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)

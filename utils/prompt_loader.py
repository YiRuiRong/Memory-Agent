from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from agent.memory_system import get_system_prompt


def _load_prompt(prompt_config_key: str, caller_name: str) -> str:
    try:
        prompt_path = get_abs_path(prompts_conf[prompt_config_key])
    except KeyError as e:
        logger.error(f"[{caller_name}]在yaml配置项中没有{prompt_config_key}配置项")
        raise e

    try:
        return open(prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[{caller_name}]解析提示词出错，{str(e)}")
        raise e


def load_system_prompts():
    return get_system_prompt(_load_prompt("main_prompt_path", "load_system_prompts"))


def load_rag_prompts():
    return _load_prompt("rag_summarize_prompt_path", "load_rag_prompts")


def load_report_prompts():
    return get_system_prompt(_load_prompt("report_prompt_path", "load_report_prompts"))


if __name__ == '__main__':
    print(load_report_prompts())


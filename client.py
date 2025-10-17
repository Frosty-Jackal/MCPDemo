# -*- coding: utf-8 -*-
"""
MCP Client (CLI)
- If --image is provided, call tool with image_path directly (NO bytes/base64).
- Streams OCR text summary via Ollama.
"""
from __future__ import annotations

import os
import re
import sys
import json
import uuid
import asyncio
import logging
import argparse
from typing import Optional, List, Dict, Any
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

try:
    import ollama
except Exception as e:
    raise RuntimeError("未找到 ollama Python 包，请先 `pip install ollama`") from e

# -------------------------
# Logging
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = (BASE_DIR / "temp").resolve(); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ocr_server.log"


def _build_logger(name: str, request_id: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s -  %(filename)s:%(lineno)d - %(request_id)s - %(message)s"
        ))
        logger.addHandler(ch)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8"); fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s -  %(filename)s:%(lineno)d - %(request_id)s - %(message)s"
        ))
        logger.addHandler(fh)

    class _Adapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return msg, {"extra": {"request_id": request_id}}
    return _Adapter(logger, {})


# -------------------------
# LLM helpers
# -------------------------

def build_path_extract_prompt(user_text: str) -> List[Dict[str, str]]:
    system = "你是一个文件路径抽取器。请从用户输入中抽取 Windows 或 Linux 本地文件路径，只输出 JSON。"
    user = (
        f"用户输入：{user_text}\n"
        "请输出JSON：{\"paths\": [\"...\"], \"reason\": \"...\"}；若无路径则 paths 为空数组。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_ocr_summary_prompt(text_results: List[str]) -> List[Dict[str, str]]:
    print(text_results)
    system = (
        "你是一名资深 OCR 结果整理与信息抽取编辑。"
        "目标：将嘈杂的 OCR 片段去噪、合并、去重、纠正常见识别错误，并按信息层级结构化为 Markdown。"
        "要求："
        "1) 不臆测：对不确定内容用『[不确定]』或留空，不自行编造。"
        "2) 保真：保留原始数字、单位、时间、专有名词；必要时统一格式（如日期 YYYY-MM-DD、金额保留2位小数）。"
        "3) 去噪：移除页眉页脚、页码、重复水印、无意义碎片；合并被错误断行/连字的句子（如 '电\n话'→'电话'、'fi-\nle'→'file'）。"
        "4) 去重：相似度高的重复项合并；保留最完整版本，并记录差异。"
        "5) 纠错：仅做保守纠错（易混字符如 O/0, l/1, B/8, '—'/'-'）；遇到公司/人名等专名若不确定不要改写。"
        "6) 结构化：优先提取标题/日期/编号/地址/电话/邮件/金额/表格/列表等关键字段，无法归类的放在“未归类”。"
        "7) 输出：仅输出 Markdown（中文），不要附带解释或额外前后缀。"
    )

    user = (
        "下面是 OCR 文本结果列表（可能重复、断行混乱、含错别字）：\n"
        f"{json.dumps(text_results, ensure_ascii=False, indent=2)}\n\n"
        "请将其整理为 Markdown\n"
        
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ollama_stream_chat(messages: List[Dict[str, str]], model: str = "qwen3:1.7b"):
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            yield chunk["message"]["content"]


def ollama_chat_once(messages: List[Dict[str, str]], model: str = "qwen3:1.7b") -> str:
    resp = ollama.chat(model=model, messages=messages)
    return resp.get("message", {}).get("content", "")


# -------------------------
# MCP Client
# -------------------------
class MCPClient:
    def __init__(self, logger: logging.LoggerAdapter):
        self.session: Optional[ClientSession] = None
        self.stdio = None
        self.write = None
        self.logger = logger
        self._exit_stack = AsyncExitStack()   # ✅ 新增

    async def connect_to_server(self, server_script_path: str):
        self.logger.info("准备连接 MCP 服务：%s", server_script_path)
        if not (server_script_path.endswith('.py') or server_script_path.endswith('.js')):
            raise ValueError("MCP 服务脚本必须是 .py 或 .js 文件")
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        self.stdio, self.write = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        resp = await self.session.list_tools()
        tools = [t.name for t in resp.tools]
        self.logger.info("已连接到 MCP 服务。可用工具：%s", tools)

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session:
            raise RuntimeError("MCP 会话未初始化")
        self.logger.info("调用工具：%s，参数：%s", tool_name, {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in args.items()})
        try:
            result = await self.session.call_tool(tool_name, arguments=args)
        except Exception as e:
            raise RuntimeError(f"调用工具失败（客户端/传输阶段）：{e}") from e

        # unwrap: result.content → parts
        content = getattr(result, "content", None)
        if content is None:
            return result if isinstance(result, dict) else {"raw": result}
        payload: Dict[str, Any] | None = None
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype == "json" and isinstance(getattr(part, "json", None), dict):
                payload = part.json
                break
            if ptype == "text" and isinstance(getattr(part, "text", None), str):
                try:
                    payload = json.loads(part.text)
                except Exception:
                    payload = {"raw_text": part.text}
                break
        if payload is None:
            payload = {"raw": repr(content[0]) if content else None}
        return payload

    async def close(self):
        await self._exit_stack.aclose()   # ✅ 统一释放两个上下文
        self.logger.info("关闭 MCP 客户端资源")



# -------------------------
# Backend once
# -------------------------
async def backend_once(server_script_path: str, query: str, image_path: Optional[str], logger: logging.LoggerAdapter) -> str:
    client = MCPClient(logger)
    await client.connect_to_server(server_script_path)

    try:
        used_tool = None
        ocr_res: Dict[str, Any] = {}

        if image_path and os.path.exists(image_path):
            logger.info("检测到图像路径，直接调用路径工具：%s", image_path)
            ocr_res = await client.call_tool("get_ppocr_result_by_image_path", {"image_path": image_path})
            used_tool = "get_ppocr_result_by_image_path"
        else:
            logger.info("未提供图像路径，尝试从 query 中解析")
            msgs = build_path_extract_prompt(query)
            llm = ollama_chat_once(msgs)
            paths: List[str] = []
            try:
                data = json.loads(llm); paths = data.get("paths", []) or []
            except Exception:
                import re
                paths = re.findall(r'([A-Za-z]:\\[^\s]+|\B/[^ \n\t]+)', llm)
            valid = next((p for p in paths if os.path.exists(p)), None)
            if valid:
                logger.info("解析到可用路径：%s", valid)
                ocr_res = await client.call_tool("get_ppocr_result_by_image_path", {"image_path": valid})
                used_tool = "get_ppocr_result_by_image_path"

        # 如果调用了 OCR 工具
        if used_tool:
            status = ocr_res.get("status")
            logger.info("工具返回：%s", status)
            if status not in {"success", "partial_success", "no_text"}:
                raise RuntimeError(f"OCR 工具错误：{ocr_res}")

            text_results = ocr_res.get("text_results") or []
            if isinstance(text_results, str):
                text_results = [text_results]
            if not text_results:
                logger.warning("OCR 没有识别到文本（status=%s）", status)
                return "# 识别结果为空\n\n未从图片中识别到可用文本。请检查图片是否清晰/包含文本。"

            # 让 LLM 总结
            md_msgs = build_ocr_summary_prompt(text_results)
            md_collect: List[str] = []
            for piece in ollama_stream_chat(md_msgs):
                sys.stdout.write(piece); sys.stdout.flush(); md_collect.append(piece)
            return "".join(md_collect)

        # 普通对话
        chat_msgs = [
            {"role": "system", "content": "你是一个专业助理，请用 Markdown 组织你的回答。"},
            {"role": "user", "content": query}
        ]
        md_collect: List[str] = []
        for piece in ollama_stream_chat(chat_msgs):
            sys.stdout.write(piece); sys.stdout.flush(); md_collect.append(piece)
        return "".join(md_collect)
    finally:
        await client.close()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Client（路径模式）")
    parser.add_argument("server_script", type=str, help="MCP 服务脚本路径（如 service.py）")
    parser.add_argument("--image", type=str, default="", help="本地图像路径（优先使用）")
    parser.add_argument("--query", type=str, default="", help="文本描述（可包含路径）")
    args = parser.parse_args()

    request_id = uuid.uuid4().hex[:8]
    logger = _build_logger("ocr_client", request_id)
    logger.info("启动 MCP Client（纯后端模式）")
    logger.info("参数：server_script=%s, image=%r, query_len=%d", args.server_script, args.image, len(args.query))

    try:
        md = asyncio.run(backend_once(args.server_script, args.query, args.image or None, logger))
        print("\n\n===== 最终 Markdown 输出 =====\n\n" + md)
    except Exception as e:
        logger.error("运行失败：%s", e, exc_info=True)
        sys.exit(2)
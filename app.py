# -*- coding: utf-8 -*-
"""
Streamlit OCR 前端（精调视觉版 · 标题更大 · 模块卡片化）
- 标题：30px 大号标题，去除顶部多余空白
- 模块：卡片化 + 更大的模块标题（18px），间距和阴影更清晰
- 左：对话卡 / 输入卡；右：设置卡 / 结果图像卡 / 中间结果卡 / 日志卡
- 逻辑沿用你现有的 MCPClient / ollama_* 等
"""
from __future__ import annotations

import os
import io
import json
import uuid
import base64
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import streamlit as st
from PIL import Image

# === 你的依赖（保持不变） ===
from client import (
    _build_logger, MCPClient, build_path_extract_prompt,
    build_ocr_summary_prompt, ollama_stream_chat, ollama_chat_once
)

# -------------------- 基础设置 --------------------
st.set_page_config(page_title="智能对话与识别助手", layout="wide", initial_sidebar_state="collapsed")
BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = (BASE_DIR / "temp").resolve(); LOG_DIR.mkdir(parents=True, exist_ok=True)
SERVER_LOG = (BASE_DIR / "temp" / "ocr_server.log")

DEFAULTS = {
    "MCP_SERVICE_PATH": os.environ.get("MCP_SERVICE_PATH", str((BASE_DIR / "service.py").resolve())),
    "OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "qwen3:1.7b"),
    "STREAM_SUMMARY": True,
    "SHOW_TOKEN_TICKER": True,
    "MAX_SHOW_OCR_LINES": 30,
}

# -------------------- 小工具 --------------------
def decode_b64_to_pil(image_b64: str) -> Optional[Image.Image]:
    try:
        if not image_b64:
            return None
        img_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

def _new_logger():
    if "_request_id" not in st.session_state:
        st.session_state["_request_id"] = uuid.uuid4().hex[:8]
    request_id = st.session_state["_request_id"]
    logger = _build_logger("ocr_app", request_id)
    return logger, request_id

def read_log_tail(path: Path, max_lines: int = 400) -> str:
    if not path.exists():
        return "(日志文件不存在)"
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:])
    except Exception as e:
        return f"(读取日志失败：{e})"

# -------------------- 样式（加大标题、收紧上边距、卡片化模块） --------------------
def inject_style():
    st.markdown("""
    <style>
      :root{
        --bg-page:#f6f7fb;
        --bg-card:#ffffff;
        --border:#e6e8ee;
        --shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 8px rgba(16,24,40,.06);
        --radius:16px;
        --bubble-user:#eaf2ff;
        --bubble-assistant:#f3f4f6;
        --text-muted:#6b7280;
      }
      /* 页顶留白缩小 */
      .block-container{ padding-top:6px !important; background:var(--bg-page); }

      /* 大标题容器（非 sticky，避免遮挡） */
      .page-hero{
        background:var(--bg-card);
        border:1px solid var(--border);
        border-radius:var(--radius);
        padding:12px 16px;
        box-shadow:var(--shadow);
        margin-bottom:12px;
      }
      .page-hero h1{
        margin:0; padding:0;
        font-size:30px; line-height:38px; font-weight:800;
      }
      .page-hero .sub{
        margin-top:2px; color:var(--text-muted); font-size:13px;
      }

      /* 通用卡片（模拟模块独立） */
      .card{
        background:var(--bg-card);
        border:1px solid var(--border);
        border-radius:var(--radius);
        box-shadow:var(--shadow);
        padding:14px 16px;
        margin-bottom:14px;
      }
      .card-title{
        font-size:18px; font-weight:700; margin:0 0 10px 0;
      }
      .card-subtle{ color:var(--text-muted); font-size:12px; margin-top:-6px; margin-bottom:10px; }

      /* 对话气泡 */
      .bubble{
        border-radius:12px; padding:10px 12px; margin:8px 0;
        border:1px solid var(--border);
        word-wrap:break-word; word-break:break-word; white-space:pre-wrap;
      }
      .user-msg{ background:var(--bubble-user); }
      .assistant-msg{ background:var(--bubble-assistant); }

      /* 代码块/长文本不截断 */
      .stCode > div{ max-height:340px; overflow:auto; }
      .stMarkdown, .stText, .stTextArea, .element-container, .stCaption, .stCode {
        word-wrap: break-word; word-break: break-word; white-space: pre-wrap;
      }

      /* 优化按钮和输入行距 */
      .stButton > button{ height:40px; }
      label.css-16idsys, .st-emotion-cache-1jicfl2, .st-emotion-cache-ue6h4q { font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

# -------------------- 业务流程（沿用原逻辑） --------------------
def run_once_with_mcp(
    server_script_path: str,
    user_text: str,
    upload_image_bytes: Optional[bytes],
    model_name: str,
    stream_summary: bool,
    show_token_ticker: bool,
    logger: logging.LoggerAdapter,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    render_markdown_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    async def _inner() -> Dict[str, Any]:
        if progress_cb: progress_cb("连接服务", 1, 5)
        client = MCPClient(logger)
        await client.connect_to_server(server_script_path)

        used_tool = None
        ocr_res: Dict[str, Any] = {}
        tmp_path: Optional[str] = None
        try:
            # Step 1: 准备输入
            if progress_cb: progress_cb("准备输入", 2, 5)
            if upload_image_bytes:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(upload_image_bytes); tmp.flush(); tmp.close()
                tmp_path = tmp.name
                logger.info("上传图像已缓存到临时文件：%s", tmp_path)
                # Step 2: OCR
                if progress_cb: progress_cb("OCR 识别", 3, 5)
                ocr_res = await client.call_tool("get_ppocr_result_by_image_path", {"image_path": tmp_path})
                used_tool = "get_ppocr_result_by_image_path"
            else:
                logger.info("未上传图像，尝试从文本解析文件路径")
                resp = ollama_chat_once(build_path_extract_prompt(user_text), model=model_name)
                try:
                    paths = (json.loads(resp).get("paths") or [])
                except Exception:
                    import re
                    paths = re.findall(r'([A-Za-z]:\\[^\s]+|\B/[^ \n\t]+)', resp)
                valid = next((p for p in paths if os.path.exists(p)), None)
                if valid:
                    logger.info("解析到有效路径：%s", valid)
                    if progress_cb: progress_cb("OCR 识别", 3, 5)
                    ocr_res = await client.call_tool("get_ppocr_result_by_image_path", {"image_path": valid})
                    used_tool = "get_ppocr_result_by_image_path"

            status = ocr_res.get("status") if used_tool else None
            text_results = ocr_res.get("text_results") if used_tool else []
            if isinstance(text_results, str):
                text_results = [text_results]

            md_text = ""
            token_count = 0
            if used_tool and text_results:
                if progress_cb: progress_cb("生成总结", 4, 5)
                md_msgs = build_ocr_summary_prompt(text_results)
                if stream_summary:
                    md_collect: List[str] = []
                    for piece in ollama_stream_chat(md_msgs, model=model_name):
                        md_collect.append(piece)
                        token_count += 1
                        md_text = "".join(md_collect)
                        if render_markdown_cb:
                            render_markdown_cb(md_text)
                else:
                    md_text = "".join(ollama_stream_chat(md_msgs, model=model_name))
                    if render_markdown_cb:
                        render_markdown_cb(md_text)
            elif used_tool and status == "no_text":
                md_text = "## 识别结果为空\n未从图片中识别到可用文本。"
                if render_markdown_cb:
                    render_markdown_cb(md_text)

            if progress_cb: progress_cb("完成", 5, 5)
            return {
                "md_text": md_text,
                "ocr_image_b64": ocr_res.get("image_base64", "") if used_tool else "",
                "used_tool": used_tool,
                "text_results": text_results or [],
                "status": status,
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.info("已删除临时文件：%s", tmp_path)
                except Exception as e:
                    logger.warning("删除临时文件失败：%s", e)
            await client.close()

    return asyncio.run(_inner())

# -------------------- 气泡渲染 --------------------
def bubble_user(text: str):
    st.markdown(f'<div class="bubble user-msg">{text}</div>', unsafe_allow_html=True)

def bubble_assistant(md_text: str):
    st.markdown(f'<div class="bubble assistant-msg">', unsafe_allow_html=True)
    st.markdown(md_text or "_无内容_", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- 主界面 --------------------
def main():
    inject_style()
    logger, request_id = _new_logger()

    # 标题（更大字号且无多余空白）
    st.markdown(
        f"""
        <div class="page-hero">
          <h1>智能对话与识别助手</h1>
          <div class="sub">会话 ID：{request_id}</div>
        </div>
        """, unsafe_allow_html=True
    )

    # 布局：左（对话/输入卡） | 右（设置/结果等卡）
    left, right = st.columns([7, 5], gap="large")

    # ---------- 左侧 ----------
    with left:
        # 对话卡
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗨️ 对话</div>', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            history = st.session_state.get("_history", [])
            for role, content in history[-8:]:
                if role == "user":
                    bubble_user(content)
                else:
                    bubble_assistant(content)
        assistant_placeholder = st.empty()
        progress_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)  # /card

        # 输入卡
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">✍️ 输入</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1], vertical_alignment="top")
        with c1:
            user_text = st.text_area("你的问题（可包含本地文件路径）", height=120,
                                     placeholder="例如：请识别这张图的发票并提取关键信息。也可输入本地路径 /path/to/img.jpg")
        with c2:
            upload = st.file_uploader("上传图片（可选）", type=["jpg", "jpeg", "png", "bmp", "webp"])
        go_col, clear_col = st.columns([1, 1])
        with go_col:
            run_btn = st.button("🚀 发送", type="primary", use_container_width=True)
        with clear_col:
            clear_btn = st.button("🧹 清空", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)  # /card

    # ---------- 右侧（卡片化的 Expander） ----------
    with right:
        # 设置卡
        with st.expander("⚙️ 设置", expanded=True):
            st.markdown('<div class="card-subtle">模型与服务路径</div>', unsafe_allow_html=True)
            service_path = st.text_input("MCP 服务脚本路径", value=DEFAULTS["MCP_SERVICE_PATH"])
            model_name = st.text_input("LLM 模型名称", value=DEFAULTS["OLLAMA_MODEL"])
            stream_summary = st.toggle("流式生成总结", value=DEFAULTS["STREAM_SUMMARY"])
            show_token_ticker = st.toggle("展示流式片段计数", value=DEFAULTS["SHOW_TOKEN_TICKER"])
            max_ocr_lines = st.number_input("中间结果：OCR 行数展示上限", min_value=5, max_value=300, step=5,
                                            value=DEFAULTS["MAX_SHOW_OCR_LINES"])

        # 结果图像卡
        with st.expander("🖼️ 结果图像（OCR 标注）", expanded=True):
            st.markdown('<div class="card-subtle">服务端返回的标注图</div>', unsafe_allow_html=True)
            img_slot = st.empty()

        # 中间结果卡
        with st.expander("🛠️ 中间结果（摘录）", expanded=False):
            st.markdown('<div class="card-subtle">OCR 文本（最多显示上限）</div>', unsafe_allow_html=True)
            mid_slot = st.empty()

        # 日志卡
        with st.expander("🧾 服务端日志（尾部）", expanded=False):
            st.markdown('<div class="card-subtle">temp/ocr_server.log</div>', unsafe_allow_html=True)
            st.code(str(SERVER_LOG), language="text")
            col_dl, col_rf = st.columns(2)
            with col_dl:
                log_bytes = SERVER_LOG.read_bytes() if SERVER_LOG.exists() else b""
                st.download_button("⬇️ 下载日志", data=log_bytes, file_name="ocr_server.log", mime="text/plain")
            with col_rf:
                if st.button("🔄 刷新日志"):
                    st.session_state["_refresh_log"] = True
            st.code(read_log_tail(SERVER_LOG, 220), language="text")

    # ---------- 交互 ----------
    if clear_btn:
        for k in ["_last_stream_chunk", "_progress_value", "_result_cache", "_history"]:
            st.session_state.pop(k, None)
        st.rerun()

    if run_btn:
        logger.info("收到前端请求，开始执行")
        # 将用户消息写入历史 + 展示
        st.session_state.setdefault("_history", []).append(("user", user_text or "(空)"))
        bubble_user(user_text or "(空)")

        def progress_cb(phase: str, step: int, total: int):
            pct = {"连接服务": 0.15, "准备输入": 0.35, "OCR 识别": 0.6, "生成总结": 0.85, "完成": 1.0}.get(phase, step / max(1, total))
            progress_placeholder.info(f"{int(pct*100)}% · 当前阶段：**{phase}**（{step}/{total}）")

        def render_markdown_cb(md_text: str):
            assistant_placeholder.markdown(f'<div class="bubble assistant-msg">{md_text}</div>', unsafe_allow_html=True)

        try:
            img_bytes = upload.read() if upload is not None else None
            result = run_once_with_mcp(
                service_path,
                user_text or "",
                img_bytes,
                model_name=model_name,
                stream_summary=stream_summary,
                show_token_ticker=show_token_ticker,
                logger=logger,
                progress_cb=progress_cb,
                render_markdown_cb=render_markdown_cb,
            )
            st.session_state["_result_cache"] = result

            final_md = result.get("md_text") or "_无内容_"
            assistant_placeholder.markdown(f'<div class="bubble assistant-msg">{final_md}</div>', unsafe_allow_html=True)
            st.session_state["_history"].append(("assistant", final_md))
            progress_placeholder.success("✅ 完成")

            # 右侧回填
            if result.get("ocr_image_b64"):
                pil_img = decode_b64_to_pil(result["ocr_image_b64"])
                if pil_img:
                    img_slot.image(pil_img, caption="OCR 结果图", use_container_width=True)
                else:
                    img_slot.warning("返回了结果图，但解析失败。")
            else:
                img_slot.info("本次未返回结果图。")

            tr = result.get("text_results") or []
            if tr:
                excerpt = "\n".join([
                    line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
                    for line in tr[:DEFAULTS["MAX_SHOW_OCR_LINES"]]
                ])
                mid_slot.code(excerpt, language="text")
            else:
                mid_slot.info("无 OCR 文本结果。")

        except Exception as e:
            logger.error("前端执行失败：%s", e, exc_info=True)
            assistant_placeholder.markdown(
                f'<div class="bubble assistant-msg">❌ 执行失败：{e}</div>', unsafe_allow_html=True
            )
            progress_placeholder.error("❌ 失败")

    # 未点击发送但已有结果时，右侧面板保持显示
    if "_result_cache" in st.session_state and not run_btn:
        result = st.session_state["_result_cache"]
        with right:
            if result.get("ocr_image_b64"):
                pil_img = decode_b64_to_pil(result["ocr_image_b64"])
                if pil_img:
                    img_slot.image(pil_img, caption="OCR 结果图", use_container_width=True)
            tr = result.get("text_results") or []
            if tr:
                excerpt = "\n".join([
                    line if isinstance(line, str) else json.dumps(line, ensure_ascii=False)
                    for line in tr[:DEFAULTS["MAX_SHOW_OCR_LINES"]]
                ])
                mid_slot.code(excerpt, language="text")

if __name__ == "__main__":
    main()

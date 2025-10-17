# -*- coding: utf-8 -*-
"""
MCP Service exposing one tool that takes an image *path* and runs PaddleOCR.
- 使用 ocr.predict(image_path)（若该签名不兼容再回退 ocr.predict(image_path, cls=True) / ocr.ocr）
- 解析 predict 结果对象（.res）中的 rec_texts/rec_scores/rec_polys/...
- 保存标注图与 JSON 到 temp/predict_{request_id}/ 目录，并将标注图回传为 base64
- 返回结构化字段：status/error/message/request_id + text_results/detailed_results/image_base64/save_dir
"""
from __future__ import annotations

import os
import io
import uuid
import json
import asyncio
import base64
import traceback
import logging
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional

from mcp.server.fastmcp import FastMCP
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np  # new

def json_sanitize(obj):
    """把返回对象中所有 numpy / Path / bytes 等不可 JSON 的类型递归转换为可序列化类型。"""
    # numpy 标量
    if isinstance(obj, (np.generic,)):
        return obj.item()
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # 路径
    if isinstance(obj, Path):
        return str(obj)
    # 二进制
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    # 容器
    if isinstance(obj, dict):
        return {k: json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_sanitize(v) for v in obj]
    # 其他原生类型原样返回
    return obj

# -------------------------
# Server & OCR init
# -------------------------
mcp = FastMCP("OCR")

# 初始化 OCR（按你实际需要调整）
ocr = PaddleOCR(use_doc_orientation_classify=True, use_doc_unwarping=True, lang="ch")

# -------------------------
# Logging
# -------------------------
BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = (BASE_DIR / "temp").resolve()
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ocr_server.log"

logger = logging.getLogger("ocr_server")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(request_id)s - %(message)s"
    ))
    logger.addHandler(ch)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(request_id)s - %(message)s"
    ))
    logger.addHandler(fh)

class RequestLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra["request_id"] = self.extra.get("request_id", "-")
        kwargs["extra"] = extra
        return msg, kwargs

logger.info("Service started. Log file → %s", str(LOG_FILE))

# -------------------------
# Helpers
# -------------------------

def _ensure_list(x):
    """把 numpy 数组等转 Python list；其他保持原样。"""
    try:
        # numpy.ndarray 有 .tolist()
        return x.tolist()  # type: ignore[attr-defined]
    except Exception:
        return x

def _parse_predict_results(results: Any, log: RequestLoggerAdapter) -> Tuple[List[Dict], List[str]]:
    """
    解析 PaddleOCR.predict 返回的结果对象列表。
    期望每个元素 r 有属性 .res（dict），包含：
      - rec_texts: List[str]
      - rec_scores: np.ndarray/list，同长度
      - rec_polys: np.ndarray(list)[N,4,2] 或 rec_boxes/dt_polys
    返回：
      parsed: List[{"bbox": [[x,y]x4] 或 [x1,y1,x2,y2], "text": str, "score": float|None}]
      texts:  仅文本列表
    """
    parsed: List[Dict] = []
    texts: List[str] = []

    if results is None:
        return parsed, texts

    # predict 通常返回 list（单页图就是 len==1）
    if not isinstance(results, list):
        # 有些版本可能直接给一个对象
        results = [results]

    for idx, r in enumerate(results):
        try:
            rd = getattr(r, "res", None)
            if rd is None and isinstance(r, dict):
                rd = r.get("res") or r  # 容错：有的例子直接 print 出 {'res': {...}}

            if not isinstance(rd, dict):
                log.warning("结果项 #%d 缺少 res 字段，类型=%s", idx, type(r).__name__)
                continue

            rec_texts = rd.get("rec_texts") or []
            rec_scores = rd.get("rec_scores") or []
            # 优先使用 rec_polys（识别框），退化用 rec_boxes/dt_polys
            rec_polys = rd.get("rec_polys") or rd.get("rec_boxes") or rd.get("dt_polys") or []

            # 转 Python list
            rec_texts = list(rec_texts)
            rec_scores = _ensure_list(rec_scores)
            rec_polys = _ensure_list(rec_polys)

            n = len(rec_texts)
            # 某些情况下 rec_scores/rec_polys 可能更长或更短，做安全下标
            for i in range(n):
                text = rec_texts[i]
                score = None
                bbox = None
                try:
                    score = float(rec_scores[i]) if i < len(rec_scores) else None
                except Exception:
                    score = None
                try:
                    bbox = rec_polys[i] if i < len(rec_polys) else None
                except Exception:
                    bbox = None

                parsed.append({"bbox": bbox, "text": text, "score": score})
                if text:
                    texts.append(text)

        except Exception:
            log.exception("解析 predict 结果时出错（项 #%d）", idx)

    return parsed, texts

async def _run_predict_by_path(image_path: str, log: RequestLoggerAdapter) -> Any:
    """
    调用 ocr.predict/ocr（兼容不同版本的签名），优先 predict(path)。
    """
    try:
        log.debug("调用 PaddleOCR.predict: %s", image_path)
        try:
            return await asyncio.to_thread(ocr.predict, image_path)  # 常见签名
        except TypeError:
            # 部分版本可能需要 cls=True
            return await asyncio.to_thread(ocr.predict, image_path)
    except AttributeError:
        # 旧版本回退到 ocr.ocr
        log.debug("predict 不可用，回退到 ocr.ocr")
        return await asyncio.to_thread(ocr.ocr, image_path, cls=True)

def _save_predict_artifacts(results: Any, save_dir: Path, log: RequestLoggerAdapter) -> List[Path]:
    """
    使用结果对象自带的 save_to_img/save_to_json 将标注图和 JSON 落盘到 save_dir。
    返回生成的标注图路径列表（可能多页）。
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_images: List[Path] = []

    if results is None:
        return saved_images

    if not isinstance(results, list):
        results = [results]

    for idx, r in enumerate(results):
        try:
            # 只有新版本的 result 对象才有这些方法
            if hasattr(r, "save_to_json"):
                r.save_to_json(str(save_dir))
            if hasattr(r, "save_to_img"):
                r.save_to_img(str(save_dir))
        except Exception:
            log.exception("保存第 #%d 个结果的文件失败", idx)

    # 粗略搜一波输出图（不同版本命名不同，这样更稳）
    for p in save_dir.glob("*.jpg"):
        saved_images.append(p)
    for p in save_dir.glob("*.png"):
        saved_images.append(p)

    # 去重 & 排序
    saved_images = sorted(set(saved_images))
    return saved_images

def _encode_image_to_b64(p: Path) -> str:
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# -------------------------
# Tool
# -------------------------
@mcp.tool()
async def get_ppocr_result_by_image_path(image_path: str) -> Dict[str, Any]:
    """
    输入：图片路径（str）
    流程：
      1) PIL.verify() 早期校验文件是否为图片
      2) 调用 OCR 的 predict/ocr
      3) 解析 predict 的 .res 字段（rec_texts/rec_scores/rec_polys）
      4) 将结果对象自带的标注图/JSON 保存到 temp/predict_{request_id}/
      5) 返回结构化 JSON，并附带第一张标注图的 base64（若存在）
    """
    request_id = uuid.uuid4().hex[:8]
    log = RequestLoggerAdapter(logger, {"request_id": request_id})
    log.info("收到 OCR 请求（路径）：%s", image_path)

    if not image_path:
        msg = "empty image_path"
        log.warning(msg)
        return {"status": "invalid_input", "error": "empty_image_path", "message": msg, "request_id": request_id}
    if not os.path.exists(image_path):
        msg = f"file not found: {image_path}"
        log.warning(msg)
        return {"status": "invalid_input", "error": "file_not_found", "message": msg, "request_id": request_id}

    # 先快速校验是图片文件
    try:
        await asyncio.to_thread(lambda p: Image.open(p).verify(), image_path)
        log.info("基本图像校验通过（PIL.verify）")
    except Exception as e:
        log.error("图像校验失败：%s", e, exc_info=True)

        return {
            "status": "invalid_image",
            "error": "image_verify_failed",
            "message": str(e),
            "trace": traceback.format_exc(),
            "request_id": request_id,
        }

    # 运行 OCR（predict 优先）
    try:
        results = await _run_predict_by_path(image_path, log)
    except Exception as e:
        log.exception("OCR 执行失败")
        return {
            "status": "ocr_error",
            "error": "ocr_execution_failed",
            "message": str(e),
            "trace": traceback.format_exc(),
            "request_id": request_id,
        }

    # 解析结果
    parsed, texts = _parse_predict_results(results, log)
    if not parsed:
        log.warning("OCR 返回为空：%s", image_path)

    # 保存标注图/JSON
    save_dir = LOG_DIR / f"predict_{request_id}"
    saved_imgs = _save_predict_artifacts(results, save_dir, log)

    # 选择一张图回传（如果有）
    image_b64 = ""
    if saved_imgs:
        try:
            image_b64 = _encode_image_to_b64(saved_imgs[0])
            log.info("返回标注图：%s", str(saved_imgs[0]))
        except Exception:
            log.exception("编码标注图失败（将继续返回文本结果）")

    # 组织响应
    status = "success" if parsed else "no_text"
    payload = {
        "status": status,
        "request_id": request_id,
        "ocr_count": len(parsed),
        "text_results": texts,          # 纯文本列表
        "detailed_results": parsed,     # 每条包含 bbox/text/score
        "image_base64": image_b64,      # 若保存了标注图则带回
        "save_dir": str(save_dir),      # 方便你落盘定位
    }
    return json_sanitize(payload)


# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")

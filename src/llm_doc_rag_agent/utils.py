from __future__ import annotations

import hashlib  # 导入 Python 标准库 hashlib ，用于作哈希计算
import json     # 导入 json ，负责 JSON 序列化 和 反序列化
from dataclasses import asdict, is_dataclass    # 从 dataclass 库中导入 asdict()-把 dataclass 对象转成普通字典  is_dataclass()-判断一个对象是否为 dataclass
from pathlib import Path
from typing import Any

# 给一段文本生成稳定的 SHA-256 Hash Value
def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() # str -> b"str" -> 用 SHA-256 创建一个哈希对象 -> 把哈希结果转换成十六进制 str

# 先生成完整哈希，再取前 length 个字符      用于需要短 ID 的地方
def short_hash(text: str, length: int = 16) -> str:
    return stable_hash(text)[:length]

# 把长文本变成一段安全、紧凑、不会太长的摘要片段
def safe_snippet(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())    # 不传参数的 .split() 会按任意空白字符切分字符串，空格、换行都会被处理掉，处理后用单个空格拼接
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."    # .retrip() 去掉右侧空白，把长文本截断到指定长度附近并用省略号结尾

# 把复杂 Python 对象转换为 JSON 可以处理的对象
def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):     # 判断 value 是否为 dataclass
        return asdict(value)
    if isinstance(value, Path): # 判断 value 是否为 Path 类型
        return str(value)
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]  # 递归处理，列表中的每个元素都再次调用 to_jsonable()    list 可能是一个嵌套 Path、dataclass、dict 的列表
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return value

# 参数：输出文件路径 要写入的数据列表   本函数不返回有意义的值，只是写文件用 
def dump_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)  # 写文件前先确保父目录存在，已存在则不报错
    with path.open("w", encoding="utf-8") as fh:    # 安全打开，用 Path 打开文件，并用 utf-8 编码写文件，把打开后的文件对象命名为 fh
        for row in rows:    # to_jsonable(row)  把当前这一条数据转成 JSON 可处理的形状  json.dumps() 把 Python 对象转为 JSON 字符串
            fh.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")   # 一条条写入文件

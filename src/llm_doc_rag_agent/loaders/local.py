"""
RAG 入库前的 本地文档安全加载层
把 哪些文件能读、哪些文件必须跳过、怎么保留来源信息、怎么给后续增量索引提供 hash 边界集中在一个地方处理
"""
from __future__ import annotations  # 类型注解延迟解析

import fnmatch  # 用来匹配类似 .ragignore 里的通用配置规则
import json     # 服务于 .ipynb ，notebook 本质上是一个 JSON 文件
from pathlib import Path    # 处理路径，比手写字符串拼接安全清楚
from typing import Iterable # 可叠戴对象，用于 _iter_fiels() ，用 yield 一个个吐出文件路径

from llm_doc_rag_agent.schemas import Document
from llm_doc_rag_agent.utils import stable_hash # 给文档内容生成稳定哈希，用于判断内容是否改变，避免重复索引

# 本地文档加载
class LocalDocumentLoader:
    """Load explicitly provided local technical documents."""

    DEFAULT_EXTENSIONS = {".md", ".markdown", ".txt", ".py", ".yaml", ".yml", ".ipynb", ".rst", ".pdf"}  # 默认允许入库的文件类型
    DEFAULT_IGNORE_PATTERNS = {                                                         # 默认忽略规则，忽略敏感文件和生成物缓存
        ".env",
        "*.env",
        "*.key",
        "*.pem",
        "*.crt",
        "*.p12",
        "*.pfx",
        "auth.json",
        ".DS_Store",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        "data/indexes/",
        "experiments/",
    }

    def __init__(
        self,
        extensions: set[str] | None = None, # 允许的文件后缀
        ignore_file: str = ".ragignore",    # 自定义忽略文件
        ignore_roots: list[str | Path] | None = None,   # 额外从哪些根目录读取忽略规则
        max_file_size_mb: int = 20, # 最大文件大小
    ) -> None:
        self.extensions = {ext.lower() for ext in (extensions or self.DEFAULT_EXTENSIONS)}  # 处理文件后缀 -> lower()
        self.ignore_file = ignore_file
        self.ignore_roots = [Path(root).expanduser().resolve() for root in (ignore_roots or [])]    # 同一成为（绝对）路径对象
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024   # MB -> bytes

    def load_path(self, path: str | Path) -> list[Document]:    # 接收字符串路径 or Path，返回多个 Document
        root = Path(path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Input path does not exist: {root}")
        if root.is_file():
            if self._should_skip_file(root, root.parent, self._load_ignore_patterns(root.parent)):
                return []
            return [self._load_file(root)]
        ignore_patterns = self._load_ignore_patterns(root)
        docs: list[Document] = []
        for file_path in sorted(self._iter_files(root, ignore_patterns)):
            docs.append(self._load_file(file_path))
        return [doc for doc in docs if doc.text.strip()]    # 过滤空文档
    # 递归遍历文件
    def _iter_files(self, root: Path, ignore_patterns: set[str]) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and not self._should_skip_file(path, root, ignore_patterns):
                yield path
    # 过滤法则
    def _should_skip_file(self, path: Path, root: Path, ignore_patterns: set[str]) -> bool:
        if path.suffix.lower() not in self.extensions:  # 后缀不在允许列表中就跳过
            return True
        relative = path.relative_to(root)   # 把绝对路径转为相对 root 路径
        if any(part.startswith(".") for part in relative.parts):    # 任意一段以 . 开头的目录/文件名就跳过
            return True
        if path.stat().st_size > self.max_file_size_bytes:  # 文件大小超过限制就跳过
            return True
        return self._matches_ignore(self._matching_relative_paths(path, root), ignore_patterns)
    # 加载 .ragignore
    def _load_ignore_patterns(self, root: Path) -> set[str]:
        patterns = set(self.DEFAULT_IGNORE_PATTERNS)    # 复制一份默认忽略规则
        ignore_dirs = [root, *self.ignore_roots]
        for ignore_dir in ignore_dirs:
            ignore_path = ignore_dir / self.ignore_file
            if ignore_path.exists() and ignore_path.is_file():
                for line in ignore_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    pattern = line.strip()
                    if pattern and not pattern.startswith("#"):
                        patterns.add(pattern)
        return patterns
    # 多根路径匹配
    def _matching_relative_paths(self, path: Path, root: Path) -> list[Path]:
        relatives = [path.relative_to(root)]
        for ignore_root in self.ignore_roots:
            if path == ignore_root or ignore_root in path.parents:
                relatives.append(path.relative_to(ignore_root))
        return relatives
    # 执行 ignore 匹配
    def _matches_ignore(self, relatives: list[Path], ignore_patterns: set[str]) -> bool:
        for pattern in ignore_patterns:
            normalized = pattern.strip().lstrip("/")
            if not normalized:
                continue
            for relative in relatives:
                relative_text = relative.as_posix()
                if normalized.endswith("/"):
                    directory = normalized.rstrip("/")
                    if relative_text.startswith(f"{directory}/") or directory in relative.parts:
                        return True
                    continue
                if fnmatch.fnmatch(relative_text, normalized) or fnmatch.fnmatch(relative.name, normalized):
                    return True
        return False
    # 读取文件
    def _load_file(self, path: Path) -> Document:
        suffix = path.suffix.lower()    # 文件后缀
        stat = path.stat()              # 文件元信息
        if suffix == ".ipynb":
            text = self._load_notebook(path)
            kind = "notebook"
        elif suffix == ".pdf":
            text = self._load_pdf(path)
            kind = "pdf"
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            kind = suffix.lstrip(".") or "text" # 把 .md 变成 md ，如果没有后缀就兜底为 text 
        return Document(    # 将原始文本和元数据封装为统一的 Document
            text=text,
            source_path=str(path),
            metadata={
                "file_type": kind,
                "file_name": path.name,
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "document_hash": stable_hash(text.strip()),
            },
        )

    def _load_notebook(self, path: Path) -> str:
        data = json.loads(path.read_text(encoding="utf-8")) # 把 .ipynb 文件从 JSON 字符串转成 Python 字典
        blocks: list[str] = []
        for index, cell in enumerate(data.get("cells", [])):
            cell_type = cell.get("cell_type", "unknown")
            source = "".join(cell.get("source", []))
            if source.strip():
                blocks.append(f"[{cell_type} cell {index}]\n{source.strip()}")  # 故意给每个 cell 加标题，进入到 RAG 后可以检索到某段内容来自哪个 cell
        return "\n\n".join(blocks)

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF loading requires optional dependency: pypdf") from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]    # 逐页提取文本
        return "\n\n".join(pages)

from __future__ import annotations

from functools import lru_cache # 导入装饰器，让函数的结果被缓存
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field  # 导入 Field ，用来个类字段设置默认值、别名、校验信息等
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):   # BaseSettings 可以让 Pydantic 自动读取 .env 内容，读取系统环境变量并覆盖值，自动做类型转化（str -> int）
    """Runtime settings loaded from env vars and optional YAML config."""

    model_config = SettingsConfigDict(  # 给 类变量 赋值
        env_file=".env",                # 读取当前目录下的 .env
        env_file_encoding="utf-8",      # 使用 utf-8 读取
        extra="ignore",                 # 忽略 环境或输入 中这个类定义不存在的字段
        env_prefix="",                  # 环境变量不额外加统一前缀
    )
    # 路径类配置
    project_root: Path = Field(default=Path("."), alias="LLM_DOC_RAG_PROJECT_ROOT") # 字段名:类型 = Field(默认值,字段对应的环境变量名)
    qdrant_path: Path = Field(default=Path("data/indexes/qdrant"), alias="LLM_DOC_RAG_QDRANT_PATH")
    default_collection: str = Field(default="llm_doc_rag", alias="LLM_DOC_RAG_COLLECTION") # alias 自动把环境变量名映射到 Python 属性名 
    # LLM 配置
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    llm_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")

    embedding_provider: str = Field(default="sentence_transformers", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=5, alias="TOP_K")
    retriever_type: str = Field(default="dense", alias="RETRIEVER_TYPE")
    candidate_k: int | None = Field(default=None, alias="CANDIDATE_K")
    reranker_model: str | None = Field(default=None, alias="RERANKER_MODEL")
    reranker_device: str = Field(default="cpu",alias="RERANKER_DEVICE")
    eval_retrievers: list[str] = Field(default_factory=lambda: ["dense"], alias="EVAL_RETRIEVERS")
    max_rewrites: int = Field(default=1, alias="MAX_REWRITES")
    max_generation_retries: int = Field(default=1, alias="MAX_GENERATION_RETRIES")
    min_relevance_score: float = Field(default=0.05, alias="MIN_RELEVANCE_SCORE")
    min_relevant_chunks: int = Field(default=1, alias="MIN_RELEVANT_CHUNKS")
    min_grounded_overlap: float = Field(default=0.2, alias="MIN_GROUNDED_OVERLAP")
    quality_grader: str = Field(default="hybrid", alias="QUALITY_GRADER")
    quality_model: str | None = Field(default=None, alias="QUALITY_MODEL")
    quality_base_url: str | None = Field(default=None, alias="QUALITY_BASE_URL")
    quality_api_key: str | None = Field(default=None, alias="QUALITY_API_KEY")
    quality_max_tokens: int = Field(default=4096, alias="QUALITY_MAX_TOKENS")
    quality_top_p: float = Field(default=0.1, alias="QUALITY_TOP_P")
    quality_disable_thinking: bool = Field(default=True, alias="QUALITY_DISABLE_THINKING")
    run_ragas: bool = Field(default=False, alias="RUN_RAGAS")
    ragas_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        alias="RAGAS_METRICS",
    )
    ragas_model: str | None = Field(default=None, alias="RAGAS_MODEL")
    ragas_base_url: str | None = Field(default=None, alias="RAGAS_BASE_URL")
    ragas_api_key: str | None = Field(default=None, alias="RAGAS_API_KEY")
    ragas_max_tokens: int = Field(default=4096, alias="RAGAS_MAX_TOKENS")
    ragas_top_p: float = Field(default=0.1, alias="RAGAS_TOP_P")
    ragas_disable_thinking: bool = Field(default=True, alias="RAGAS_DISABLE_THINKING")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    @property   # 装饰器，封装 解析路径 这个计算函数————访问 settings.function 就像访问普通属性，但背后做了路径的解析       本身不存解析结果，但是每次访问都要计算
    def resolved_project_root(self) -> Path:
        return self.project_root.expanduser().resolve() # self.project_root 取当前配置对象中的路径字段  .expanduser() 把 ~ 展开成用户目录   .resolve() 转成解析后的绝对路径

    @property
    def resolved_qdrant_path(self) -> Path:
        path = self.qdrant_path.expanduser()
        if path.is_absolute():
            return path                             # 绝对路径
        return self.resolved_project_root / path    # 项目根目录 / 相对路径

    # 读取 YAML 的方法
    def with_yaml(self, config_path: str | Path | None) -> "Settings":
        if not config_path:
            return self
        path = Path(config_path).expanduser()   # 把传入的路径包装成 Path ，再展开 ~
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}   # 用 utf-8 读取文件文本，并把 YAML 文本解析成 Python 数据
        return self.model_copy(update=_normalize_yaml_keys(data))   # 先把 YAML 数据整理成合法字段，再复制当前 Pydantic 模型，并用 update 中的值覆盖原值

# 普通内部辅助函数，不希望外部直接调用
def _normalize_yaml_keys(data: dict[str, Any]) -> dict[str, Any]:
    aliases = { # 白名单
        # "输入键":"内部字段名"
        "project_root": "project_root",
        "qdrant_path": "qdrant_path",
        "default_collection": "default_collection",
        "collection": "default_collection", # YAML 中如果写 collection ，内部会映射到 default_collection
        "embedding_provider": "embedding_provider",
        "embedding_model": "embedding_model",
        "embedding_device": "embedding_device",
        "chunk_size": "chunk_size",
        "chunk_overlap": "chunk_overlap",
        "top_k": "top_k",
        "retriever_type": "retriever_type",
        "candidate_k": "candidate_k",
        "reranker_model": "reranker_model",
        "reranker_device": "reranker_device",
        "eval_retrievers": "eval_retrievers",
        "max_rewrites": "max_rewrites",
        "max_generation_retries": "max_generation_retries",
        "min_relevance_score": "min_relevance_score",
        "min_relevant_chunks": "min_relevant_chunks",
        "min_grounded_overlap": "min_grounded_overlap",
        "quality_grader": "quality_grader",
        "quality_model": "quality_model",
        "quality_base_url": "quality_base_url",
        "quality_api_key": "quality_api_key",
        "quality_max_tokens": "quality_max_tokens",
        "quality_top_p": "quality_top_p",
        "quality_disable_thinking": "quality_disable_thinking",
        "run_ragas": "run_ragas",
        "ragas_metrics": "ragas_metrics",
        "ragas_model": "ragas_model",
        "ragas_base_url": "ragas_base_url",
        "ragas_api_key": "ragas_api_key",
        "ragas_max_tokens": "ragas_max_tokens",
        "ragas_top_p": "ragas_top_p",
        "ragas_disable_thinking": "ragas_disable_thinking",
        "llm_model": "llm_model",
        "llm_base_url": "llm_base_url",
    }
    normalized = {
        aliases[k]: v
        for k, v in data.items()
        if k in aliases and not _is_empty_yaml_value(v)
    }   # 只保留白名单中的键，把 YAML key 转成内部字段名
    if isinstance(normalized.get("eval_retrievers"), str):  # 判断这个 key 的 value 是否为 str
        normalized["eval_retrievers"] = [   # 把用逗号分隔的字符串转成列表
            item.strip()
            for item in normalized["eval_retrievers"].split(",")
            if item.strip()
        ]
    if isinstance(normalized.get("ragas_metrics"), str):
        normalized["ragas_metrics"] = [
            item.strip()
            for item in normalized["ragas_metrics"].split(",")
            if item.strip()
        ]
    for key in ("project_root", "qdrant_path"):
        if key in normalized:
            normalized[key] = Path(normalized[key])
    return normalized   # 返回整理好的字典


def _is_empty_yaml_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())

# 缓存配置对象，单例模式，全局只有一个 Settings 实例
@lru_cache(maxsize=1)   # 装饰器，最多缓存 1 个结果，让下面的函数只执行一次，第一次调用时创建 Settings 并缓存，后续的调用都是直接返回缓存的同一个对象
def get_settings() -> Settings:
    return Settings()

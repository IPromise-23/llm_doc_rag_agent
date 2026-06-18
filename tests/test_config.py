from pathlib import Path

from llm_doc_rag_agent.config import Settings


def test_default_config_compares_all_core_retrievers():
    settings = Settings().with_yaml(Path("configs/default.yaml"))

    assert settings.eval_retrievers == ["dense", "bm25", "hybrid_rrf", "hybrid_rerank"]


def test_empty_yaml_values_do_not_override_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/test")
    config = tmp_path / "config.yaml"
    config.write_text("reranker_model:\n", encoding="utf-8")

    settings = Settings().with_yaml(config)

    assert settings.reranker_model == "cross-encoder/test"


def test_settings_load_retrieval_options_from_yaml(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "retriever_type: hybrid_rrf",
                "candidate_k: 12",
                "reranker_model: test-reranker",
                "eval_retrievers: dense,bm25",
                "max_rewrites: 2",
                "min_relevance_score: 0.2",
                "min_relevant_chunks: 2",
                "min_grounded_overlap: 0.3",
                "quality_grader: hybrid",
                "quality_model: judge-model",
                "quality_base_url: https://judge.example/v1",
                "quality_api_key: judge-key",
                "quality_max_tokens: 2048",
                "quality_top_p: 0.2",
                "quality_disable_thinking: false",
                "run_ragas: true",
                "ragas_metrics: faithfulness,answer_relevancy",
                "ragas_model: ragas-judge",
                "ragas_base_url: https://ragas.example/v1",
                "ragas_api_key: ragas-key",
                "ragas_max_tokens: 1024",
                "ragas_top_p: 0.3",
                "ragas_disable_thinking: false",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings().with_yaml(config)

    assert settings.retriever_type == "hybrid_rrf"
    assert settings.candidate_k == 12
    assert settings.reranker_model == "test-reranker"
    assert settings.eval_retrievers == ["dense", "bm25"]
    assert settings.max_rewrites == 2
    assert settings.min_relevance_score == 0.2
    assert settings.min_relevant_chunks == 2
    assert settings.min_grounded_overlap == 0.3
    assert settings.quality_grader == "hybrid"
    assert settings.quality_model == "judge-model"
    assert settings.quality_base_url == "https://judge.example/v1"
    assert settings.quality_api_key == "judge-key"
    assert settings.quality_max_tokens == 2048
    assert settings.quality_top_p == 0.2
    assert settings.quality_disable_thinking is False
    assert settings.run_ragas is True
    assert settings.ragas_metrics == ["faithfulness", "answer_relevancy"]
    assert settings.ragas_model == "ragas-judge"
    assert settings.ragas_base_url == "https://ragas.example/v1"
    assert settings.ragas_api_key == "ragas-key"
    assert settings.ragas_max_tokens == 1024
    assert settings.ragas_top_p == 0.3
    assert settings.ragas_disable_thinking is False

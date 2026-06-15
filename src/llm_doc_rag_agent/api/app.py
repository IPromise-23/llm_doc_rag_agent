from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_doc_rag_agent.config import get_settings
from llm_doc_rag_agent.evaluation import EvalRunner
from llm_doc_rag_agent.service import RagService
from llm_doc_rag_agent.utils import to_jsonable


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("API server requires optional dependencies: fastapi and uvicorn") from exc

    class IngestRequest(BaseModel):
        path: str
        recreate: bool = False

    class QueryRequest(BaseModel):
        question: str
        top_k: int | None = None
        retriever_type: str | None = None
        candidate_k: int | None = None
        use_graph: bool = True

    class EvalRequest(BaseModel):
        dataset: str
        output: str | None = None
        top_k: int | None = None

    app = FastAPI(title="llm_doc_rag_agent", version="0.1.0")
    settings = get_settings()

    def service(collection: str) -> RagService:
        return RagService(settings, collection=collection)

    def allowed_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        project_root = settings.resolved_project_root
        if path == project_root or project_root in path.parents:
            return path
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "path_outside_project_root",
                "message": "Path must be inside the configured project root.",
                "details": {"project_root": str(project_root)},
            },
        )

    @app.exception_handler(Exception)
    def handle_error(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "message": str(exc),
                "details": {"request_path": request.url.path},
            },
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/collections/{collection}/ingest")
    def ingest(collection: str, request: IngestRequest) -> dict[str, Any]:
        return to_jsonable(service(collection).ingest_path(allowed_path(request.path), recreate=request.recreate))

    @app.post("/collections/{collection}/query")
    def query(collection: str, request: QueryRequest) -> dict[str, Any]:
        answer = service(collection).query(
            request.question,
            top_k=request.top_k,
            use_graph=request.use_graph,
            retriever_type=request.retriever_type,
            candidate_k=request.candidate_k,
        )
        return to_jsonable(answer)

    @app.post("/collections/{collection}/eval")
    def eval_collection(collection: str, request: EvalRequest) -> dict[str, Any]:
        dataset = allowed_path(request.dataset)
        output = allowed_path(request.output) if request.output else None
        results = EvalRunner(service(collection)).run(dataset, output_path=output, top_k=request.top_k)
        return {"examples": len(results), "output": request.output, "results": to_jsonable(results)}

    @app.get("/collections/{collection}/sources")
    def sources(collection: str, limit: int | None = 200) -> list[str]:
        return service(collection).list_sources(limit=limit)

    @app.get("/collections/{collection}/chunks")
    def chunks(collection: str, source: str, limit: int | None = 100) -> list[dict[str, Any]]:
        return to_jsonable(service(collection).chunks_for_source(source_path=str(allowed_path(source)), limit=limit))

    return app

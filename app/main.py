"""FastAPI 应用：输入版本提交、异步生成、结果查询。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query
from fastapi.responses import RedirectResponse
from sqlite3 import Connection

from . import db
from .config import DB_PATH, WORKER_THREADS
from .models import (
    CurrentMergeOut,
    CurrentResultOut,
    GenerationOut,
    HealthOut,
    MergeJobOut,
    MergeRequest,
    NewDistrictSubmission,
    OldGridSubmission,
    VersionOut,
)

# 单线程工作池：生成任务按提交顺序串行执行与发布
_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _executor
    db.init_db()
    _executor = ThreadPoolExecutor(
        max_workers=WORKER_THREADS, thread_name_prefix="generation"
    )
    yield
    _executor.shutdown(wait=True)
    _executor = None


app = FastAPI(
    title="城市统计片区重划分摊服务",
    description=(
        "整版提交旧网格（编号、整数矩形、户数）与新片区（编号、整数矩形），"
        "按重叠面积以最大余数法逐格分摊户数并守恒总户数。"
        "生成任务固定提交时刻的版本快照，生成期间可继续替换任一输入。"
        "另支持片区合并供数：对当前已发布结果按共边连通关系把片区合并为"
        "达到最低户数的最优化分组，结果绑定源生成任务并原子发布。"
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def get_db() -> Connection:
    conn = db.get_conn()
    try:
        yield conn
    finally:
        conn.close()


def _items_from_old(payload: OldGridSubmission) -> list[dict]:
    return [
        {
            "id": it.id,
            "rect": it.rect.as_tuple(),
            "households": it.households,
        }
        for it in payload.items
    ]


def _items_from_new(payload: NewDistrictSubmission) -> list[dict]:
    return [{"id": it.id, "rect": it.rect.as_tuple()} for it in payload.items]


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthOut, tags=["运维"])
def health(conn: Connection = Depends(get_db)):
    conn.execute("SELECT 1").fetchone()
    return {"status": "ok", "db": str(DB_PATH)}


@app.post("/old-grids", response_model=VersionOut, status_code=201, tags=["输入版本"])
def submit_old_grid(payload: OldGridSubmission, conn: Connection = Depends(get_db)):
    """整版提交旧网格；替换旧输入即产生新版本（历史版本保留）。"""
    try:
        with conn:
            version_id = db.insert_version("old", _items_from_old(payload), conn)
    except db.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return db.get_version(version_id, conn)


@app.post(
    "/new-districts", response_model=VersionOut, status_code=201, tags=["输入版本"]
)
def submit_new_districts(
    payload: NewDistrictSubmission, conn: Connection = Depends(get_db)
):
    """整版提交新片区；替换旧输入即产生新版本（历史版本保留）。"""
    try:
        with conn:
            version_id = db.insert_version("new", _items_from_new(payload), conn)
    except db.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return db.get_version(version_id, conn)


@app.get("/versions", response_model=list[VersionOut], tags=["输入版本"])
def list_versions(
    kind: str | None = Query(None, pattern="^(old|new)$"),
    conn: Connection = Depends(get_db),
):
    """列出输入版本，可按 kind=old/new 过滤。"""
    return db.list_versions(kind, conn)


@app.get("/versions/{version_id}", response_model=VersionOut, tags=["输入版本"])
def get_version(
    version_id: int = PathParam(..., ge=1), conn: Connection = Depends(get_db)
):
    try:
        return db.get_version(version_id, conn)
    except db.NoVersionError:
        raise HTTPException(status_code=404, detail=f"版本 {version_id} 不存在")


@app.post(
    "/generations",
    response_model=GenerationOut,
    status_code=202,
    tags=["生成任务"],
)
def create_generation(conn: Connection = Depends(get_db)):
    """开始一次生成：同步固定当前最新 old/new 版本快照，计算在后台串行执行。

    快照在本请求的事务内确定；此后替换任一输入不影响本任务。
    """
    try:
        with conn:
            gen_id = db.create_generation(conn)
    except db.NoVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    assert _executor is not None
    _executor.submit(db.run_generation, gen_id, DB_PATH)
    return db.get_generation(gen_id, conn)


@app.get("/generations", response_model=list[GenerationOut], tags=["生成任务"])
def list_generations(conn: Connection = Depends(get_db)):
    return db.list_generations(conn)


@app.get("/generations/{gen_id}", response_model=GenerationOut, tags=["生成任务"])
def get_generation(
    gen_id: int = PathParam(..., ge=1), conn: Connection = Depends(get_db)
):
    row = db.get_generation(gen_id, conn)
    if row is None:
        raise HTTPException(status_code=404, detail=f"生成任务 {gen_id} 不存在")
    return row


@app.get("/result", response_model=CurrentResultOut, tags=["发布结果"])
def get_current_result(conn: Connection = Depends(get_db)):
    """当前已发布结果（片区汇总 + 两份输入版本）。

    新结果完整发布前，这里始终返回上一份已发布结果；从未发布过时返回 404。
    """
    result = db.get_current_result(conn)
    if result is None:
        raise HTTPException(status_code=404, detail="尚无已发布的生成结果")
    return result


@app.post(
    "/merges",
    response_model=MergeJobOut,
    status_code=202,
    tags=["片区合并供数"],
)
def create_merge(payload: MergeRequest, conn: Connection = Depends(get_db)):
    """发起一次片区合并供数。

    以**当前已发布结果**的片区矩形和户数为输入，在本请求事务内固定其生成任务 id；
    后台按共边连通关系分组，使每组总户数达到 min_households，并最大化组数。
    无解时任务落为 failed 并说明原因，不发布任何部分结果。
    """
    try:
        with conn:
            job_id = db.create_merge_job(payload.min_households, conn)
    except db.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except db.NoResultError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    assert _executor is not None
    _executor.submit(db.run_merge_job, job_id, DB_PATH)
    return db.get_merge_job(job_id, conn)


@app.get("/merges", response_model=list[MergeJobOut], tags=["片区合并供数"])
def list_merge_jobs(conn: Connection = Depends(get_db)):
    """列出全部合并任务（含失败原因与 stale 标记）。"""
    return db.list_merge_jobs(conn)


@app.get("/merges/{job_id}", response_model=MergeJobOut, tags=["片区合并供数"])
def get_merge_job(
    job_id: int = PathParam(..., ge=1), conn: Connection = Depends(get_db)
):
    """查询单个合并任务；stale=true 表示源结果已被替换，该合并版不得当作新版供应。"""
    view = db.get_merge_job(job_id, conn)
    if view is None:
        raise HTTPException(status_code=404, detail=f"合并任务 {job_id} 不存在")
    return view


@app.get("/merge-result", response_model=CurrentMergeOut, tags=["片区合并供数"])
def get_current_merge(conn: Connection = Depends(get_db)):
    """当前供数的合并版（绑定其源生成任务）。

    若源结果已被新发布结果替换，响应中 stale=true：旧合并版仅作历史保留，
    不能当作新版供应；需对新结果重新发起合并，重算成功后这里才供应新版。
    从未成功合并过时返回 404。
    """
    view = db.get_current_merge(conn)
    if view is None:
        raise HTTPException(status_code=404, detail="尚无成功发布的合并供数结果")
    return view

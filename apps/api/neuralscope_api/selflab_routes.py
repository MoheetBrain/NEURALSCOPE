"""SelfLab HTTP API — everything under ``/api/selflab`` (docs/selflab/INTERFACES.md §5).

Error mapping inside this router (independent of the app-level handlers):
  400  request-body / query validation errors and invalid specs (FastAPI's default here would be 422)
  413  declared request body larger than 1 MiB
  403  mutating request from a non-loopback client without a valid token; api_only disabled
  404  unknown case / experiment / trial / run
  409  invalid state transition (run twice, cancel a finished experiment, resume with a changed plan …)
  503  the model is still loading (or failed to load) for endpoints that need it

SECURITY. Every POST endpoint requires one of:
  * a loopback client (127.0.0.0/8, ::1, ::ffff:127.0.0.1) whose ``Host`` header and (if present) ``Origin``
    header also name a loopback host — this blocks DNS-rebinding and cross-site browser requests; requests
    proxied by the local Next dev server arrive from 127.0.0.1. Starlette's in-process TestClient pseudo-hosts
    (client ``testclient``, Host ``testserver``) are accepted ONLY under pytest (``PYTEST_CURRENT_TEST`` set) or
    when the router is built with ``allow_test_hosts=True``; production rejects them;
  * or, when ``NEURALSCOPE_API_TOKEN`` is set, that token in the ``X-NeuralScope-Token`` header.
Read-only GET endpoints are not guarded (the server binds to 127.0.0.1 by default).

Inputs are bounded (message count/length, max_new_tokens ≤ 64, prompt tokens ≤ 1024, Budget limits from
``selflab/schemas.py``). Target output and documents are data only; nothing is interpreted or executed.
"""

from __future__ import annotations

import hmac
import ipaddress
import os
import threading
from typing import Any, Callable, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, ValidationError

from neuralscope.experiments.service import ServiceError
from neuralscope.selflab.export import build_export
from neuralscope.selflab.investigator import suggest as rule_suggest
from neuralscope.selflab.registry import list_templates
from neuralscope.selflab.schemas import ClaimType, ExperimentSpec
from neuralscope.selflab.service import (
    MAX_DOCUMENT_CHARS,
    MAX_HINT_CHARS,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    MAX_NEW_TOKENS,
    MAX_QUESTION_CHARS,
    MAX_RECEIPTS,
    MAX_RESPONSE_CHARS,
    MAX_SYSTEM_CHARS,
    SelfLabService,
    case_detail,
    create_api_only_case,
    status_payload,
    templates_payload,
)
from neuralscope.selflab.store import SelfLabError, SelfLabStore

MAX_BODY_BYTES = 1024 * 1024  # declared Content-Length bound for SelfLab requests (fields are bounded separately)
TOKEN_ENV = "NEURALSCOPE_API_TOKEN"
TOKEN_HEADER = "X-NeuralScope-Token"
_LOOPBACK_NAMES = {"localhost"}
_LOCAL_HOSTNAMES = {"localhost"}
# Starlette TestClient pseudo-hosts (in-process, not routable). Accepted only in test mode (see pseudo_hosts_allowed).
_TEST_CLIENT_HOSTS = {"testclient"}  # request.client.host
_TEST_HOSTNAMES = {"testserver"}  # default Host header


# ====================================================================================== request models
class _In(BaseModel):
    model_config = {"extra": "forbid"}


class MessageIn(_In):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class ChatIn(_In):
    messages: list[MessageIn] | None = Field(default=None, max_length=MAX_MESSAGES - 2)
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    document: str | None = Field(default=None, max_length=MAX_DOCUMENT_CHARS)
    hint: str | None = Field(default=None, max_length=MAX_HINT_CHARS)
    system: str | None = Field(default=None, max_length=MAX_SYSTEM_CHARS)
    max_new_tokens: int = Field(default=24, ge=1, le=MAX_NEW_TOKENS)


class CaseIn(_In):
    access_level: Literal["white_box_local", "api_only"] = "white_box_local"
    run_id: str | None = Field(default=None, max_length=64)
    messages: list[MessageIn] | None = Field(default=None, max_length=MAX_MESSAGES)
    response_text: str | None = Field(default=None, max_length=MAX_RESPONSE_CHARS)
    receipts: list[dict[str, Any]] | None = Field(default=None, max_length=MAX_RECEIPTS)
    char_start: int = Field(ge=0, le=MAX_RESPONSE_CHARS)
    char_end: int = Field(ge=1, le=MAX_RESPONSE_CHARS)
    claim_type: ClaimType | None = None
    title: str | None = Field(default=None, max_length=200)


class ExperimentIn(_In):
    spec: ExperimentSpec


class SuggestIn(_In):
    case_id: str = Field(min_length=1, max_length=64)


# ====================================================================================== error mapping
class SelfLabRoute(APIRoute):
    """Maps SelfLab errors and validation errors to JSON responses inside this router only."""

    def get_route_handler(self) -> Callable:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            try:
                n = int(request.headers.get("content-length") or 0)
            except ValueError:
                n = 0
            if n > MAX_BODY_BYTES:
                return JSONResponse(status_code=413, content={"detail": f"request body exceeds {MAX_BODY_BYTES} bytes"})
            try:
                return await original(request)
            except RequestValidationError as e:
                return JSONResponse(status_code=400, content={"detail": jsonable_encoder(e.errors())})
            except ValidationError as e:
                return JSONResponse(status_code=400, content={"detail": jsonable_encoder(
                    e.errors(include_url=False, include_context=False))})
            except SelfLabError as e:
                return JSONResponse(status_code=e.status_code, content={"detail": str(e)})
            except ServiceError as e:
                return JSONResponse(status_code=400, content={"detail": str(e)})

        return handler


# ====================================================================================== security
def pseudo_hosts_allowed(flag: bool | None = None) -> bool:
    """Explicit flag wins; otherwise only while pytest is running a test (PYTEST_CURRENT_TEST is set by pytest)."""
    return bool(flag) if flag is not None else "PYTEST_CURRENT_TEST" in os.environ


def is_loopback_host(host: str | None, allow_test: bool = False) -> bool:
    if not host:
        return False
    h = host.strip().strip("[]").split("%", 1)[0].lower()
    if h in _LOOPBACK_NAMES or (allow_test and h in _TEST_CLIENT_HOSTS):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    if ip.is_loopback:
        return True
    mapped = getattr(ip, "ipv4_mapped", None)
    return bool(mapped is not None and mapped.is_loopback)


def _is_local_hostname(hostname: str | None, allow_test: bool = False) -> bool:
    if not hostname:
        return False
    h = hostname.strip().strip("[]").lower()
    return (h in _LOCAL_HOSTNAMES or (allow_test and h in _TEST_HOSTNAMES) or h.endswith(".localhost")
            or is_loopback_host(h))


def _host_of(header_value: str) -> str | None:
    try:
        return urlsplit(header_value if "//" in header_value else f"//{header_value}").hostname
    except ValueError:
        return None


def check_local_or_token(request: Request, allow_test_hosts: bool | None = None) -> None:
    token = os.environ.get(TOKEN_ENV, "")
    presented = request.headers.get(TOKEN_HEADER, "")
    if token and presented and hmac.compare_digest(token.encode("utf-8"), presented.encode("utf-8")):
        return
    test = pseudo_hosts_allowed(allow_test_hosts)
    client = request.client.host if request.client else None
    if not is_loopback_host(client, allow_test=test):
        raise HTTPException(403, "SelfLab mutating endpoints accept loopback clients only"
                                 + (f" (or a valid {TOKEN_HEADER} header)" if token else
                                    f" (set {TOKEN_ENV} on the server to allow token-authenticated remote clients)"))
    host = request.headers.get("host")
    if host and not _is_local_hostname(_host_of(host), allow_test=test):
        raise HTTPException(403, "Host header does not name a loopback host (DNS-rebinding protection)")
    origin = request.headers.get("origin")
    if origin and origin != "null" and not _is_local_hostname(_host_of(origin), allow_test=test):
        raise HTTPException(403, "cross-origin request from a non-local Origin")
    if origin == "null":
        raise HTTPException(403, "requests with an opaque (null) Origin are refused")


def require_local_or_token(request: Request) -> None:
    """Dependency with automatic test-host mode (accepted only while pytest runs a test)."""
    check_local_or_token(request, None)


# ====================================================================================== router
def build_selflab_router(state: Any, allow_test_hosts: bool | None = None) -> APIRouter:
    """``state`` is the API's ``AppState`` (settings, jobs, service, status, error).

    ``allow_test_hosts``: accept Starlette TestClient pseudo-hosts. None (default) = only while pytest runs a test.
    """
    store = SelfLabStore(state.settings.db_path)
    holder: dict[str, Any] = {"svc": None}
    lock = threading.Lock()
    router = APIRouter(prefix="/api/selflab", tags=["selflab"], route_class=SelfLabRoute)
    def guard_dep(request: Request) -> None:
        check_local_or_token(request, allow_test_hosts)

    guard = [Depends(guard_dep)]

    def current() -> SelfLabService | None:
        if state.service is None:
            return None
        with lock:
            svc = holder["svc"]
            if svc is None or svc.ns is not state.service:
                svc = SelfLabService(state.service, store, state.jobs)
                holder["svc"] = svc
            return svc

    def sl() -> SelfLabService:
        if state.status == "error":
            raise HTTPException(503, f"model failed to load: {state.error}")
        svc = current()
        if svc is None:
            raise HTTPException(503, "model is still loading — poll /api/health")
        return svc

    router.store = store  # type: ignore[attr-defined]  # exposed for tests / introspection
    router.selflab_service = current  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ status
    @router.get("/status")
    def status() -> dict[str, Any]:
        svc = current() if state.status == "ready" else None
        return status_payload(svc, state.status, state.error, bool(getattr(state.settings, "is_demo", False)))

    # ------------------------------------------------------------------ chat
    @router.post("/chat", dependencies=guard)
    def chat(body: ChatIn, s: SelfLabService = Depends(sl)) -> dict[str, Any]:
        return s.chat(question=body.question, document=body.document, hint=body.hint, system=body.system,
                      messages=[m.model_dump() for m in body.messages] if body.messages else None,
                      max_new_tokens=body.max_new_tokens)

    # ------------------------------------------------------------------ cases
    @router.post("/cases", dependencies=guard)
    def create_case(body: CaseIn) -> dict[str, Any]:
        if body.access_level == "api_only":
            if body.run_id is not None:
                raise HTTPException(400, "api_only cases are built from pasted messages, not a run_id")
            if not body.messages or body.response_text is None:
                raise HTTPException(400, "api_only cases need `messages` and `response_text`")
            case = create_api_only_case(store, [m.model_dump() for m in body.messages], body.response_text,
                                        body.char_start, body.char_end, body.receipts, body.claim_type, body.title)
            return case.model_dump(mode="json")
        if not body.run_id:
            raise HTTPException(400, "white_box_local cases need `run_id` (from POST /api/selflab/chat)")
        if body.messages is not None or body.response_text is not None or body.receipts is not None:
            raise HTTPException(400, "white_box_local cases take messages/response/receipts from the run, not the "
                                     "request body")
        case = sl().create_case(body.run_id, body.char_start, body.char_end, body.claim_type, body.title)
        return case.model_dump(mode="json")

    @router.get("/cases")
    def list_cases(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        return [c.model_dump(mode="json") for c in store.list_cases(limit=limit, offset=offset)]

    @router.get("/cases/{case_id}")
    def get_case(case_id: str) -> dict[str, Any]:
        return case_detail(store, store.require_case(case_id))

    # ------------------------------------------------------------------ templates
    @router.get("/templates")
    def templates(case_id: str | None = Query(None, max_length=64)) -> list[dict[str, Any]]:
        return templates_payload(store, case_id)

    # ------------------------------------------------------------------ experiments
    @router.post("/experiments", dependencies=guard)
    def commit_experiment(body: ExperimentIn, s: SelfLabService = Depends(sl)) -> dict[str, Any]:
        return s.commit(body.spec).model_dump(mode="json")

    @router.get("/experiments")
    def list_experiments(case_id: str | None = Query(None, max_length=64), limit: int = Query(100, ge=1, le=500),
                         offset: int = Query(0, ge=0)) -> list[dict[str, Any]]:
        return [e.model_dump(mode="json") for e in store.list_experiments(case_id=case_id, limit=limit,
                                                                          offset=offset)]

    @router.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return store.require_experiment(experiment_id).model_dump(mode="json")

    @router.post("/experiments/{experiment_id}/run", dependencies=guard)
    def run_experiment(experiment_id: str, s: SelfLabService = Depends(sl)) -> dict[str, Any]:
        store.require_experiment(experiment_id)
        return s.engine.run(experiment_id).model_dump(mode="json")

    @router.post("/experiments/{experiment_id}/cancel", dependencies=guard)
    def cancel_experiment(experiment_id: str) -> dict[str, Any]:
        svc = current()
        if svc is None:  # nothing can be running without a model; only 'committed' can be cancelled
            rec = store.require_experiment(experiment_id)
            if rec.status != "committed":
                raise HTTPException(503, "model is still loading — poll /api/health")
            rec = store.transition(experiment_id, {"committed"}, status="cancelled",
                                   message="cancelled before it was queued (no trials ran)")
            store.log_event(experiment_id, "cancelled", {"before_queue": True})
            return rec.model_dump(mode="json")
        return svc.engine.cancel(experiment_id).model_dump(mode="json")

    @router.post("/experiments/{experiment_id}/resume", dependencies=guard)
    def resume_experiment(experiment_id: str, s: SelfLabService = Depends(sl)) -> dict[str, Any]:
        store.require_experiment(experiment_id)
        return s.engine.resume(experiment_id).model_dump(mode="json")

    @router.get("/experiments/{experiment_id}/trials")
    def list_trials(experiment_id: str, response: Response, offset: int = Query(0, ge=0),
                    limit: int = Query(200, ge=1, le=5000)) -> list[dict[str, Any]]:
        store.require_experiment(experiment_id)
        response.headers["X-Total-Count"] = str(len(store.trial_indices(experiment_id)))
        return [t.model_dump(mode="json") for t in store.list_trials(experiment_id, offset=offset, limit=limit)]

    @router.get("/experiments/{experiment_id}/events")
    def list_events(experiment_id: str) -> list[dict[str, Any]]:
        store.require_experiment(experiment_id)
        return store.list_events(experiment_id)

    @router.get("/experiments/{experiment_id}/export")
    def export_experiment(experiment_id: str, preview: bool = Query(False),
                          exclude_free_text: bool = Query(False)) -> dict[str, Any]:
        return build_export(store, experiment_id, preview=preview, exclude_free_text=exclude_free_text)

    # ------------------------------------------------------------------ trials
    @router.get("/trials/{trial_id}")
    def get_trial(trial_id: str) -> dict[str, Any]:
        t = store.require_trial(trial_id)
        return {**t.model_dump(mode="json"), "lab_runs": store.trial_runs(trial_id)}

    @router.post("/trials/{trial_id}/open-in-lab", dependencies=guard)
    def open_in_lab(trial_id: str, s: SelfLabService = Depends(sl)) -> dict[str, Any]:
        store.require_trial(trial_id)
        return s.open_in_lab(trial_id)

    # ------------------------------------------------------------------ evidence
    @router.get("/evidence")
    def evidence(case_id: str | None = Query(None, max_length=64),
                 experiment_id: str | None = Query(None, max_length=64)) -> list[dict[str, Any]]:
        if case_id:
            store.require_case(case_id)
        if experiment_id:
            store.require_experiment(experiment_id)
        return [e.model_dump(mode="json") for e in store.list_evidence(case_id=case_id, experiment_id=experiment_id)]

    # ------------------------------------------------------------------ investigator
    @router.post("/investigate/suggest", dependencies=guard)
    def investigate_suggest(body: SuggestIn) -> dict[str, Any]:
        return rule_suggest(store.require_case(body.case_id), list_templates())

    return router

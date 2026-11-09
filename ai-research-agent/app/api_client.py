"""Small HTTP client used by the Streamlit UI to talk to the FastAPI backend."""
from __future__ import annotations

import httpx


class ApiError(RuntimeError):
    pass


class Api:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _request(self, method: str, path: str, *, timeout: float = 30, **kwargs) -> httpx.Response:
        try:
            resp = httpx.request(method, self.base + path, timeout=timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise ApiError(f"Cannot reach the backend at {self.base}. Is it running? ({exc})") from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise ApiError(str(detail))
        return resp

    def health(self) -> dict: return self._request("GET", "/health", timeout=5).json()
    def stats(self) -> dict: return self._request("GET", "/stats").json()
    def tools(self) -> list: return self._request("GET", "/tools", timeout=60).json()
    def documents(self) -> list: return self._request("GET", "/documents").json()
    def delete_document(self, doc_id: int) -> None: self._request("DELETE", f"/documents/{doc_id}", timeout=60)
    def runs(self) -> list: return self._request("GET", "/runs").json()
    def run(self, run_id: int) -> dict: return self._request("GET", f"/runs/{run_id}").json()
    def evaluations(self) -> list: return self._request("GET", "/evaluations").json()

    def upload(self, name: str, data: bytes, mime: str = "application/octet-stream") -> dict:
        return self._request("POST", "/documents", timeout=600, files={"file": (name, data, mime)}).json()

    def start_research(self, question: str, style: str, evaluate: bool) -> int:
        return self._request("POST", "/research", json={"question": question, "style": style,
                                                         "evaluate": evaluate}).json()["run_id"]

    def chat(self, session_id: str, message: str, evaluate: bool = True) -> int:
        return self._request("POST", "/chat", json={"session_id": session_id, "message": message,
                                                     "evaluate": evaluate}).json()["run_id"]

    def chat_history(self, session_id: str) -> list:
        return self._request("GET", f"/chat/{session_id}").json()

    def evaluate_run(self, run_id: int) -> dict:
        return self._request("POST", f"/runs/{run_id}/evaluate", timeout=180).json()

    def report(self, run_id: int, fmt: str) -> bytes:
        return self._request("GET", f"/runs/{run_id}/report", params={"format": fmt}, timeout=60).content

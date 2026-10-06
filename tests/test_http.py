import pytest
from digitaledge import http


class _Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b, self.headers = code, body or {}, {}

    def json(self):
        return self._b

    def raise_for_status(self):
        if self.status_code >= 400:
            raise http.requests.HTTPError(str(self.status_code))


def test_404_fails_fast_without_retrying(monkeypatch):
    calls = []
    monkeypatch.setattr(http._session, "get", lambda *a, **k: calls.append(1) or _Resp(404))
    monkeypatch.setattr(http.time, "sleep", lambda s: pytest.fail("must not back off on a 404"))
    with pytest.raises(http.NotRetryable):
        http.get_json("https://x.test/a", cache=False)
    assert len(calls) == 1


def test_429_is_retried_then_succeeds(monkeypatch):
    seq = [_Resp(429), _Resp(200, {"ok": 1})]
    monkeypatch.setattr(http._session, "get", lambda *a, **k: seq.pop(0))
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.get_json("https://x.test/b", cache=False) == {"ok": 1}

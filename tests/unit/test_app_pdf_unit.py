import importlib
import sys
import types


class _DummyContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummySessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


def _make_stub_streamlit_module():
    st = types.ModuleType("streamlit")
    st.session_state = _DummySessionState()
    st.sidebar = _DummyContext()

    st.set_page_config = lambda **kwargs: None
    st.markdown = lambda *args, **kwargs: None
    st.title = lambda *args, **kwargs: None
    st.caption = lambda *args, **kwargs: None
    st.header = lambda *args, **kwargs: None
    st.text_input = lambda *args, **kwargs: ""
    st.divider = lambda *args, **kwargs: None
    st.columns = lambda *args, **kwargs: (_DummyContext(), _DummyContext())
    st.download_button = lambda *args, **kwargs: None
    st.chat_input = lambda *args, **kwargs: None
    st.chat_message = lambda *args, **kwargs: _DummyContext()
    st.status = lambda *args, **kwargs: _DummyContext()
    st.error = lambda *args, **kwargs: None
    st.stop = lambda: None
    return st


def _import_app_with_stubbed_streamlit(monkeypatch):
    """Target file: app.py; provides access to convert_markdown_to_pdf."""
    stub_st = _make_stub_streamlit_module()
    monkeypatch.setitem(sys.modules, "streamlit", stub_st)
    if "app" in sys.modules:
        del sys.modules["app"]
    return importlib.import_module("app")


def test_convert_markdown_to_pdf_returns_bytes_and_sanitizes_script(monkeypatch):
    """Target: app.convert_markdown_to_pdf in app.py."""
    app_module = _import_app_with_stubbed_streamlit(monkeypatch)

    captured = {}

    class _PisaStatus:
        err = 0

    def fake_create_pdf(html, dest):
        captured["html"] = html
        dest.write(b"%PDF-1.4 test")
        return _PisaStatus()

    monkeypatch.setattr(app_module.pisa, "CreatePDF", fake_create_pdf)

    pdf_bytes = app_module.convert_markdown_to_pdf("Hello <script>alert('x')</script> world")

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert "<script" not in captured["html"].lower()


def test_convert_markdown_to_pdf_returns_none_on_pdf_error(monkeypatch):
    """Target: app.convert_markdown_to_pdf in app.py."""
    app_module = _import_app_with_stubbed_streamlit(monkeypatch)

    class _PisaStatus:
        err = 1

    def fake_create_pdf(_html, dest):
        return _PisaStatus()

    monkeypatch.setattr(app_module.pisa, "CreatePDF", fake_create_pdf)

    pdf_bytes = app_module.convert_markdown_to_pdf("# Title")
    assert pdf_bytes is None

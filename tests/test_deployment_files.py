import ast
import os
import re
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
IMPORT_TO_PACKAGE = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
INSTALLED_ELSEWHERE = {"torch": "installed from the CPU wheel index in the Dockerfile",
                       "shap": "optional, imported lazily inside try/except"}


def _requirement_names(path: Path):
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[<>=\[ ]", line)[0].lower())
    return names


def _third_party_imports():
    std = set(sys.stdlib_module_names)
    found = set()
    for folder in ("src", "api", "app"):
        for p in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return {m for m in found if m not in std and m not in ("src", "api", "app")}


def test_every_runtime_import_is_installed_in_the_image():
    reqs = _requirement_names(ROOT / "requirements-docker.txt")
    missing = [m for m in sorted(_third_party_imports())
               if m not in INSTALLED_ELSEWHERE and IMPORT_TO_PACKAGE.get(m, m).lower() not in reqs]
    assert missing == [], f"imported by the app but missing from requirements-docker.txt: {missing}"


def test_docker_requirements_exclude_test_and_heavy_dev_tools():
    reqs = _requirement_names(ROOT / "requirements-docker.txt")
    assert not ({"pytest", "httpx", "shap", "torch"} & reqs)
    assert {"fastapi", "uvicorn", "streamlit", "xgboost"} <= reqs


def test_dockerfile_basics():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert text.startswith("# ") and "FROM python:3.12" in text
    assert "download.pytorch.org/whl/cpu" in text and "libgomp1" in text
    assert "requirements-docker.txt" in text and "USER app" in text
    assert "EXPOSE 8000 8501" in text and "${PORT:-8000}" in text
    assert "COPY .streamlit" in text                      # dark theme config ships in the image


def test_compose_file_wires_api_and_dashboard():
    cfg = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api, dash = cfg["services"]["api"], cfg["services"]["dashboard"]
    assert "8000:8000" in api["ports"] and "8501:8501" in dash["ports"]
    assert dash["depends_on"]["api"]["condition"] == "service_healthy"
    for svc in (api, dash):
        assert any(v.startswith("./models:") for v in svc["volumes"])
        assert any(v.startswith("./reports:") for v in svc["volumes"])
    assert "healthcheck" in api


def test_ci_workflow_and_ignore_files():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    assert set(wf["jobs"]) == {"test", "docker"} and wf["jobs"]["docker"]["needs"] == "test"
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert {".venv", ".git", "data", "models"} <= set(ignore)
    assert "Dockerfile text eol=lf" in (ROOT / ".gitattributes").read_text(encoding="utf-8")


def test_bootstrap_trains_only_when_models_are_missing():
    from src.serving.bootstrap import ensure_models, main

    calls = []

    def fake_train(ticker):
        calls.append(ticker)
        for task in ("classification", "regression"):
            (Path(d) / f"{ticker}_xgb_{task}.joblib").write_text("x")

    with tempfile.TemporaryDirectory() as d:
        assert ensure_models("aapl", d, fake_train) is True and calls == ["AAPL"]
        assert ensure_models("AAPL", d, fake_train) is False and calls == ["AAPL"]


def test_bootstrap_is_a_no_op_without_auto_train():
    from src.serving.bootstrap import main

    os.environ.pop("AUTO_TRAIN", None)
    assert main() == 0

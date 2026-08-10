"""Public bilingual documentation contract for the RISE release surface."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TITLE = "RISE: Adaptive Imagination for World Action Models"
CHAIN = "P0 -> Field -> Calibration -> P1 -> Stop -> Oracle -> Gate"
PUBLIC_DIFFUSION_REFERENCE_URLS = (
    "https://arxiv.org/abs/2212.09748",
    "https://arxiv.org/abs/2011.13456",
    "https://arxiv.org/abs/2211.01095",
)

PUBLIC_DOCUMENTS = (
    Path("README.md"),
    Path("README_zh-CN.md"),
    Path("docs/reproduction.md"),
    Path("docs/reproduction_zh-CN.md"),
    Path("docs/configuration.md"),
    Path("configs/train/navsim/cvoi_manual_full/README.md"),
    Path("configs/eval/navsim/cvoi_manual_epdms/README.md"),
)

# These are created by the already-approved Task 4. Task 3 must link to them,
# but must not create placeholder governance files merely to satisfy link checks.
TASK4_PENDING_RELATIVE_LINKS = frozenset(
    {
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
    }
)

TRAINING_CONFIGS = (
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
)
HANDOFF_SUFFIXES = (
    "handoff/p0_selected.pt",
    "handoff/field.pt",
    "handoff/calibration.pt",
    "handoff/p1_selected.pt",
    "handoff/stop.pt",
    "handoff/oracle_full.sqlite3",
    "handoff/gate.pt",
)
REQUIRED_EPDMS_ENVIRONMENT = (
    "OPENSCENE_DATA_ROOT",
    "NAVSIM_EXP_ROOT",
    "NUPLAN_MAPS_ROOT",
    "NAVSIM_DEVKIT_ROOT",
)

_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
_EXTERNAL_LINK = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_H2_HEADING = re.compile(r"^##\s+", re.MULTILINE)
_ARXIV_URL = re.compile(r"https?://(?:www\.)?arxiv\.org/[^\s<>)\]]+", flags=re.IGNORECASE)
_PRIVATE_IP = re.compile(
    "".join(
        (
            r"\b(?:10(?:\.\d{1,3}){3}",
            r"|192\.168(?:\.\d{1,3}){2}",
            r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b",
        )
    )
)


def _read(relative_path: Path) -> str:
    path = REPOSITORY_ROOT / relative_path
    assert path.is_file(), f"missing public document: {relative_path.as_posix()}"
    return path.read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    assert match is not None, f"missing section {heading!r}"
    remainder = text[match.end() :]
    next_heading = _H2_HEADING.search(remainder)
    return remainder if next_heading is None else remainder[: next_heading.start()]


def _normalized_arxiv_urls(text: str) -> set[str]:
    normalized: set[str] = set()
    for match in _ARXIV_URL.finditer(text):
        url = match.group(0).rstrip(".,;:")
        origin = re.match(r"https?://(?:www\.)?arxiv\.org", url, flags=re.IGNORECASE)
        assert origin is not None
        normalized.add("https://arxiv.org" + url[origin.end() :])
    return normalized


def _relative_link_targets(relative_path: Path, text: str) -> set[str]:
    targets: set[str] = set()
    for match in _MARKDOWN_LINK.finditer(text):
        target = match.group("target").strip().strip("<>").split(maxsplit=1)[0]
        target = unquote(target.split("#", 1)[0])
        if not target or target.startswith("#") or _EXTERNAL_LINK.match(target):
            continue
        resolved = (REPOSITORY_ROOT / relative_path.parent / target).resolve(strict=False)
        try:
            targets.add(resolved.relative_to(REPOSITORY_ROOT).as_posix())
        except ValueError as error:
            raise AssertionError(f"{relative_path} links outside the repository: {target}") from error
    return targets


def test_required_public_document_set_exists() -> None:
    for relative_path in PUBLIC_DOCUMENTS:
        assert (REPOSITORY_ROOT / relative_path).is_file(), relative_path.as_posix()


@pytest.mark.parametrize("relative_path", [Path("README.md"), Path("README_zh-CN.md")])
def test_readmes_use_the_exact_title_two_static_badges_and_reciprocal_language_links(
    relative_path: Path,
) -> None:
    text = _read(relative_path)
    assert text.splitlines()[0] == f"# {TITLE}"
    assert len(re.findall(r"!\[[^\]]+\]\(https://img\.shields\.io/badge/", text)) == 2
    assert "License-MIT" in text
    assert re.search(r"Python-(?:%3E%3D|>=)3\.11", text)
    assert "README_zh-CN.md" in text if relative_path.name == "README.md" else "README.md" in text


@pytest.mark.parametrize(
    ("relative_path", "required_phrases"),
    [
        (
            Path("README.md"),
            (
                "Release status",
                "code and configuration only",
                "independent reproduction",
                "NavTest Full-controller EPDMS",
                "H=4",
                "forthcoming paper",
                "research software",
                "not approved for direct real-vehicle control",
            ),
        ),
        (
            Path("README_zh-CN.md"),
            (
                "发布状态",
                "仅包含代码与配置",
                "独立复现",
                "NavTest Full-controller EPDMS",
                "H=4",
                "论文发布",
                "研究软件",
                "不适用于真实车辆直接控制",
            ),
        ),
    ],
)
def test_readmes_state_release_scope_method_boundary_and_limitations(
    relative_path: Path,
    required_phrases: tuple[str, ...],
) -> None:
    text = _read(relative_path)
    for phrase in required_phrases:
        assert phrase.casefold() in text.casefold(), f"{relative_path}: missing {phrase!r}"
    assert CHAIN in text
    assert "weights" in text.casefold() or "权重" in text
    assert "Counterfactual" in text
    assert "numerical results" in text.casefold() or "数值结果" in text
    assert "not released" in text.casefold() or "未发布" in text
    assert "```mermaid" in text


def test_readmes_do_not_imply_a_research_only_license_restriction() -> None:
    english = _read(Path("README.md"))
    chinese = _read(Path("README_zh-CN.md"))
    assert "research use only" not in english.casefold()
    assert "仅限研究用途" not in chinese


@pytest.mark.parametrize(
    ("relative_path", "acknowledgements_heading", "independent_diffusion_pattern"),
    [
        (
            Path("README.md"),
            "Acknowledgements and support",
            r"(?:\bindependently\s+written\b(?:(?![.!?])[\s\S]){0,160}?\btrajectory[-\s]+diffusion\b|"
            r"\btrajectory[-\s]+diffusion\b(?:(?![.!?])[\s\S]){0,160}?\bindependently\s+written\b)",
        ),
        (
            Path("README_zh-CN.md"),
            "致谢与支持",
            r"(?:独立编写(?:(?![。！？])[\s\S]){0,120}?轨迹扩散|" r"轨迹扩散(?:(?![。！？])[\s\S]){0,120}?独立编写)",
        ),
    ],
)
def test_readmes_document_independent_trajectory_diffusion_and_its_public_technical_basis(
    relative_path: Path,
    acknowledgements_heading: str,
    independent_diffusion_pattern: str,
) -> None:
    acknowledgements = _section(_read(relative_path), acknowledgements_heading)

    assert re.search(independent_diffusion_pattern, acknowledgements, flags=re.IGNORECASE), relative_path
    assert _normalized_arxiv_urls(acknowledgements) == set(PUBLIC_DIFFUSION_REFERENCE_URLS)


def test_readmes_keep_the_rise_paper_unreleased_while_linking_only_technical_papers() -> None:
    english = _read(Path("README.md"))
    chinese = _read(Path("README_zh-CN.md"))

    english_statements = re.findall(
        r"\bRISE(?:['’]s)?\s+paper\b(?P<body>(?:(?![.!?])[\s\S]){0,160})",
        english,
        flags=re.IGNORECASE,
    )
    assert any(re.search(r"\bnot(?:\s+yet)?\s+released\b", body, flags=re.IGNORECASE) for body in english_statements)
    for body in english_statements:
        assert (
            re.search(
                r"\b(?:is|was|has\s+been)\s+(?!not\b)(?:now\s+)?"
                r"(?:published|released|publicly\s+available|available(?:\s+online|\s+on\s+arxiv)?|online|"
                r"on\s+arxiv)\b",
                body,
                flags=re.IGNORECASE,
            )
            is None
        )

    chinese_statements = re.findall(
        r"RISE\s*(?:的\s*)?论文(?P<body>(?:(?![。！？])[\s\S]){0,100})",
        chinese,
        flags=re.IGNORECASE,
    )
    assert any(re.search(r"(?:尚未发布|未发布)", body) for body in chinese_statements)
    for body in chinese_statements:
        assert (
            re.search(
                r"(?:已|已经|现已)\s*(?:发布|公开|上线|在线获取|公开可用|可在线获取|在线可用)",
                body,
            )
            is None
        )


@pytest.mark.parametrize("relative_path", [Path("README.md"), Path("README_zh-CN.md")])
def test_readmes_link_the_reproduction_configuration_and_governance_documents(relative_path: Path) -> None:
    text = _read(relative_path)
    required_targets = {
        "README.md" if relative_path.name == "README_zh-CN.md" else "README_zh-CN.md",
        "docs/configuration.md",
        "docs/reproduction.md",
        "docs/reproduction_zh-CN.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
    }
    assert required_targets <= _relative_link_targets(relative_path, text)


def test_relative_markdown_links_exist_except_for_the_explicit_task4_pending_set() -> None:
    missing: set[str] = set()
    for relative_path in PUBLIC_DOCUMENTS:
        text = _read(relative_path)
        for target in _relative_link_targets(relative_path, text):
            if not (REPOSITORY_ROOT / target).exists():
                missing.add(target)
    assert missing <= TASK4_PENDING_RELATIVE_LINKS


def test_public_documents_reject_internal_provenance_private_paths_ci_and_invented_paper_material() -> None:
    combined = "\n".join(_read(path) for path in PUBLIC_DOCUMENTS)
    forbidden_literals = (
        "/" + "disk/",
        "/home/",
        "/mnt/",
        "/workspace/",
        "172." + "16.",
        "baseline sha",
        "cvoi-main-minimal",
        "docs/superpowers",
        "experiment/cvoi",
        "fair" + "internal",
        "hch_" + "workspace",
        "MINIMAL_FILES",
        "next approved release-preparation task",
        "release/rise-public-prep",
        "future security policy",
        "下一项已批准",
        "未来安全策略",
        "world4drive",
        "github/actions",
        "actions/workflows",
        "build status",
        "doi.org",
        "@article",
        "@inproceedings",
        "BibTeX",
    )
    for marker in forbidden_literals:
        assert marker.casefold() not in combined.casefold(), marker
    assert _normalized_arxiv_urls(combined) == set(PUBLIC_DIFFUSION_REFERENCE_URLS)
    assert re.search(r"\b[0-9a-f]{40}\b", combined, flags=re.IGNORECASE) is None
    private_identity_pattern = r"\b(?:" + "he" + r"chenghao|chenghao\." + "he|cowa" + r"robot)\b"
    assert re.search(private_identity_pattern, combined, flags=re.IGNORECASE) is None
    assert re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", combined, flags=re.IGNORECASE) is None
    assert re.search(r"\b(?:release|experiment|feature|hotfix)/[a-z0-9_.-]+", combined, flags=re.IGNORECASE) is None
    assert _PRIVATE_IP.search(combined) is None
    assert re.search(r"!\[[^\]]*(?:CI|build)[^\]]*\]", combined, flags=re.IGNORECASE) is None


@pytest.mark.parametrize(
    ("relative_path", "results_heading"),
    [(Path("README.md"), "Results"), (Path("README_zh-CN.md"), "结果")],
)
def test_results_sections_contain_no_metric_values_or_benchmark_tables(
    relative_path: Path,
    results_heading: str,
) -> None:
    results = _section(_read(relative_path), results_heading)
    assert not any(line.lstrip().startswith("|") for line in results.splitlines())
    assert re.search(r"\b(?:EPDMS|PDMS|ADE|FDE)\b[^\n]*\d", results, flags=re.IGNORECASE) is None


@pytest.mark.parametrize("relative_path", [Path("docs/reproduction.md"), Path("docs/reproduction_zh-CN.md")])
def test_reproduction_guides_cover_the_exact_manual_chain_and_execution_boundaries(relative_path: Path) -> None:
    text = _read(relative_path)
    assert CHAIN in text
    for config_name in TRAINING_CONFIGS:
        assert f"configs/train/navsim/cvoi_manual_full/{config_name}" in text
    assert text.count("tools/generate_navsim_cf_trajectory_quality.py") >= 2
    assert "<chosen-p0-checkpoint>" in text
    assert "<chosen-p1-checkpoint>" in text
    assert '"/path/to/rise/results/cvoi_manual_full/p0/<chosen-p0-checkpoint>"' in text
    assert '"/path/to/rise/results/cvoi_manual_full/p1/<chosen-p1-checkpoint>"' in text
    assert "handoff/p0_selected.pt" in text
    assert "handoff/p1_selected.pt" in text
    for horizon in range(5):
        assert re.search(rf"run_cvoi_manual_oracle\.py\s+score[\s\\]+--horizon\s+{horizon}\b", text)
    assert "run_cvoi_manual_oracle.py aggregate" in text
    assert "tools/run_cvoi_direct_epdms.py" in text
    for variable_name in REQUIRED_EPDMS_ENVIRONMENT:
        assert variable_name in text
    assert "/path/to/" in text
    assert "/" + "disk/" not in text


def test_configuration_guide_documents_flat_yaml_path_groups_and_fixed_handoffs() -> None:
    text = _read(Path("docs/configuration.md"))
    for config_name in TRAINING_CONFIGS:
        assert config_name in text
    assert "configs/eval/navsim/cvoi_manual_epdms/full_controller.yaml" in text
    for group in ("data", "checkpoint", "output", "audit", "sidecar", "scene filter"):
        assert group in text.casefold()
    for suffix in HANDOFF_SUFFIXES:
        assert suffix in text
    for phrase in (
        "flat YAML",
        "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml",
        "Full results root",
        "ablation output root",
        "production preflight",
        "environment-variable interpolation",
        "inheritance",
        "DAG",
        "scheduler",
    ):
        assert phrase.casefold() in text.casefold(), phrase


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("configs/train/navsim/cvoi_manual_full/README.md"),
        Path("configs/eval/navsim/cvoi_manual_epdms/README.md"),
    ],
)
def test_config_directory_readmes_are_concise_public_indexes(relative_path: Path) -> None:
    text = _read(relative_path)
    assert len(text.splitlines()) <= 80
    assert "docs/configuration.md" in text
    assert "reproduction" in text.casefold()

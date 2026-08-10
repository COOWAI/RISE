"""License, attribution, and community-governance release contract."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIFFUSION_PATHS = (
    "app/vjepa_cowa_world_model/models/diffusion_planner.py",
    "app/vjepa_cowa_world_model/diffusion_utils/__init__.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sde.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sampling.py",
)
PUBLIC_DIFFUSION_REFERENCE_URLS = (
    "https://arxiv.org/abs/2212.09748",
    "https://arxiv.org/abs/2011.13456",
    "https://arxiv.org/abs/2211.01095",
)
_ARXIV_URL = re.compile(r"https?://(?:www\.)?arxiv\.org/[^\s<>)\]]+", flags=re.IGNORECASE)
DPM_SOLVER_MIT_LICENSE = """MIT License

Copyright (c) 2022 Cheng Lu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    assert match is not None, f"missing section {heading!r}"
    remainder = text[match.end() :]
    next_heading = re.search(r"^##\s+", remainder, flags=re.MULTILINE)
    return remainder if next_heading is None else remainder[: next_heading.start()]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _normalized_arxiv_urls(text: str) -> set[str]:
    normalized: set[str] = set()
    for match in _ARXIV_URL.finditer(text):
        url = match.group(0).rstrip(".,;:")
        origin = re.match(r"https?://(?:www\.)?arxiv\.org", url, flags=re.IGNORECASE)
        assert origin is not None
        normalized.add("https://arxiv.org" + url[origin.end() :])
    return normalized


def test_root_license_preserves_meta_and_adds_rise_notice() -> None:
    license_text = _read("LICENSE")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) Meta Platforms, Inc. and affiliates." in license_text
    assert "Copyright (c) 2026 RISE Contributors" in license_text
    assert "Permission is hereby granted, free of charge" in license_text


def test_apache_license_has_one_unambiguous_canonical_location() -> None:
    assert not (REPO_ROOT / "APACHE-LICENSE").exists()
    apache_text = _read("licenses/Apache-2.0.txt")
    assert "Apache License\n                           Version 2.0, January 2004" in apache_text
    assert not (REPO_ROOT / "_".join(("Drive", "JEPA")) / "LICENSE").exists()


def test_third_party_notices_map_every_verified_retained_component() -> None:
    notices = _read("THIRD_PARTY_NOTICES.md")
    required_tokens = (
        "Meta Platforms",
        "Facebook",
        "src/datasets/utils/worker_init_fn.py",
        "Lightning AI",
        "src/datasets/utils/video/randaugment.py",
        "src/datasets/utils/video/randerase.py",
        "Ross Wightman",
        "configs/navsim/scene_filters/navtrain.yaml",
        "configs/navsim/scene_filters/navtest.yaml",
        "licenses/Apache-2.0.txt",
        "Apache-2.0",
        "app/vjepa_cowa_world_model/diffusion_utils/dpm_solver_pytorch.py",
        "DPM-Solver",
        "Cheng Lu",
        "MIT",
    )
    for token in required_tokens:
        assert token in notices
    assert "Copyright (c) 2022 Cheng Lu" in notices
    assert "Permission is hereby granted, free of charge" in notices


def test_third_party_notices_resolve_the_rise_trajectory_diffusion_implementation() -> None:
    notices = _read("THIRD_PARTY_NOTICES.md")
    section = _section(notices, "RISE trajectory diffusion implementation")
    folded = section.casefold()

    assert "independently written rise code" in folded
    assert "root mit license" in folded
    for relative_path in PUBLIC_DIFFUSION_PATHS:
        assert relative_path in section
    assert _normalized_arxiv_urls(section) == set(PUBLIC_DIFFUSION_REFERENCE_URLS)


def test_third_party_notices_remove_the_old_unresolved_xtr_blocker_wording() -> None:
    notices = _read("THIRD_PARTY_NOTICES.md").casefold()
    forbidden_phrases = (
        "unresolved xtr-derived sources",
        "public-release blocker",
        "not asserted to be covered by rise's mit grant",
        "public-export gate must fail",
        "unresolved file remains",
    )

    for phrase in forbidden_phrases:
        assert phrase not in notices


def test_third_party_notices_preserve_the_complete_cheng_lu_mit_license() -> None:
    dpm_solver_section = _section(_read("THIRD_PARTY_NOTICES.md"), "DPM-Solver")

    assert _normalize_whitespace(DPM_SOLVER_MIT_LICENSE) in _normalize_whitespace(dpm_solver_section)


def test_verified_third_party_notices_include_auditable_upstream_urls() -> None:
    notices = _read("THIRD_PARTY_NOTICES.md")
    required_urls = (
        "https://github.com/facebookresearch/vjepa2",
        "https://github.com/huggingface/pytorch-image-models",
        "https://github.com/LuChengTHU/dpm-solver",
    )
    for url in required_urls:
        assert url in notices
    assert "a944e7744e57a5a2c13f3c73b9735edf2f71e329" in _read("src/datasets/utils/worker_init_fn.py")


def test_existing_third_party_headers_are_preserved() -> None:
    expected_lines = {
        "src/datasets/utils/worker_init_fn.py": (
            "# Copyright (c) Meta Platforms, Inc. and affiliates.",
            "# Copyright The Lightning AI team.",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
        ),
        "src/datasets/utils/video/randaugment.py": (
            "# Copyright (c) Meta Platforms, Inc. and affiliates.",
            "# Copyright 2020 Ross Wightman",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
        ),
        "src/datasets/utils/video/randerase.py": (
            "# Copyright (c) Meta Platforms, Inc. and affiliates.",
            "# Copyright 2020 Ross Wightman",
            '# Licensed under the Apache License, Version 2.0 (the "License");',
        ),
    }
    for relative_path, lines in expected_lines.items():
        source = _read(relative_path)
        for line in lines:
            assert line in source


def test_software_citation_has_no_unreleased_paper_or_repository_identity() -> None:
    citation = yaml.safe_load(_read("CITATION.cff"))
    assert citation["cff-version"] == "1.2.0"
    assert citation["type"] == "software"
    assert citation["title"] == "RISE: Adaptive Imagination for World Action Models"
    assert citation["version"] == "0.1.0"
    assert citation["license"] == "MIT"
    assert citation["authors"] == [{"name": "RISE Contributors"}]
    for prohibited in ("doi", "repository-code", "url"):
        assert prohibited not in citation


def test_public_community_and_security_files_exist_without_internal_contacts() -> None:
    required_paths = (
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CHANGELOG.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
    )
    combined = ""
    for relative_path in required_paths:
        path = REPO_ROOT / relative_path
        assert path.is_file(), relative_path
        combined += path.read_text(encoding="utf-8")
    assert "Private Vulnerability Reporting" in _read("SECURITY.md")
    assert not re.search(r"[A-Za-z0-9._%+-]+@(?!invalid\.example)[A-Za-z0-9.-]+\.[A-Za-z]{2,}", combined)
    assert not (REPO_ROOT / ".github/workflows").exists()


def test_gitignore_allows_templates_and_contains_no_machine_cache_path() -> None:
    gitignore = _read(".gitignore")
    assert ".github/" not in gitignore
    for private_marker in ("/" + "disk/", "hch_" + "workspace", "cowa" + "robot", "172." + "16."):
        assert private_marker not in gitignore

"""Documentation integrity: machine-checkable parts of the deliverables checklist.

These tests do not judge prose quality. They check the things that must never drift:
exact required strings, minimum lengths, that every generated table is current and present
in all five placements, that links resolve, that named people appear only where allowed,
that the documents stay on this repository's topic, and that nothing large or sensitive sits
in the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from experiments.sync_docs import sync

PROPOSED_ENDEAVOR = (
    "Design and implement advanced machine learning and large language model architectures "
    "for secure, reliable system-level optimization in order to enhance the efficiency, "
    "reliability, and security of U.S. digital infrastructure and accelerate the deployment of "
    "next-generation artificial intelligence (AI) applications."
)
STATUS_BANNER = (
    "Draft reference implementation — values printed in the source paper are shown for "
    "reference only; see RESULTS.md for the values this code produces and the conditions under "
    "which it produces them."
)

# Profiles that may appear in chapter 12 of the project report of this repository.
RESEARCHERS_HERE = (
    "Minxuan Zhou",
    "Binghui Wang",
    "Nathaniel Hudson",
    "Weizhao Jin",
    "Shuli Jiang",
)
# Names that must not appear anywhere in this repository.
RESEARCHERS_ELSEWHERE = ("Zhiling Lan",)
INVESTIGATION = "docs/investigations/01-federated-privacy-cross-domain.md"
DIAGRAMS = (
    "01-system-overview.md",
    "02-federated-detail.md",
    "03-trust-boundary.md",
    "04-experiment-flow.md",
)
NOTEBOOKS = ("01_federated_privacy_tradeoff.ipynb",)
CHARTS = ("federated_workclass.png", "epsilon_tradeoff.png")
# Method-level vocabulary of the companion project; a short pointer to the companion by name
# is fine, describing its method here is not.
OFF_TOPIC = re.compile(
    r"\bDDQN\b|Double DQN|\bSJF\b|\bFCFS\b|Round Robin|\bSRTF\b|\bguardrails?\b|\bturnaround\b"
)
ADULT_URL = "http://archive.ics.uci.edu/ml/datasets/Adult"
DATASET_LINE = (
    "Dataset: UCI Adult / Census Income (1994 U.S. Census extraction; 48,842 rows; "
    "workclass clients Private / Government / Self-employed plus IID control)"
)

README = "README.md"
RESULTS = "experiments/results/RESULTS.md"
REPORT = "docs/report/PROJECT_REPORT.md"
POSTER = "docs/poster/POSTER.md"
SLIDES = "docs/slides/PRESENTATION.md"
DATA_CARDS = "data/README.md"
FIVE_PLACEMENTS = (RESULTS, README, REPORT, POSTER, SLIDES)
# Leftovers of the combined project this repository was split from.
SPLIT_LEFTOVERS = re.compile(
    r"SPLIT-TODO|track1_|track2_|\bTrack [12]\b|secure-systems-optimization"
    r"|Secure Systems Optimization|[Tt]wo source papers|[Bb]oth tracks"
)
# The documents talk about the problem, the implementation, the Before / After results and the
# limits. They do not discuss what the source paper states, leaves open or prints differently.
PAPER_DISCUSSION = re.compile(
    r"\b(?:notes?|open points?|gaps?) [AB]-F?[0-9]|\b[AB]-F[0-9]\b|[Cc]onstant predictor"
    r"|leaves? open|does not specify|not specified|as printed in Paper|\bPaper [AB]\b"
    r"|confirm or dispute"
)
IGNORED_DIRS = {
    ".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "runs", "_executed",
    "adult", "cache", "femnist", "node_modules",
}  # fmt: skip

GENERATED = re.compile(
    r"<!-- BEGIN GENERATED: [\w\-]+ -->.*?<!-- END GENERATED: [\w\-]+ -->", re.DOTALL
)
FENCED = re.compile(r"```.*?```", re.DOTALL)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _read(repo_root: Path, relative: str) -> str:
    path = repo_root / relative
    assert path.is_file(), f"missing required document: {relative}"
    return path.read_text(encoding="utf-8")


def _prose_words(text: str) -> int:
    """Words of hand-written prose: generated blocks and code blocks do not count."""
    return len(FENCED.sub(" ", GENERATED.sub(" ", text)).split())


def _flat(text: str) -> str:
    """Collapse whitespace and blockquote markers so wrapped quotations compare equal."""
    return re.sub(r"\s+", " ", re.sub(r"(?m)^\s*>\s?", "", text))


def _repo_files(repo_root: Path) -> list[Path]:
    files = []
    for path in repo_root.rglob("*"):
        relative = path.relative_to(repo_root)
        if any(part in IGNORED_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def _documents(repo_root: Path) -> list[Path]:
    """Hand-written documents: Markdown, notebooks, citation and community files."""
    return [
        path
        for path in _repo_files(repo_root)
        if path.suffix in {".md", ".ipynb", ".cff", ".yml"} and "generated" not in path.parts
    ]


@pytest.mark.parametrize(
    ("relative", "minimum"), [(README, 3000), (REPORT, 8000), (INVESTIGATION, 2500)]
)
def test_long_documents_meet_their_minimum_length(
    repo_root: Path, relative: str, minimum: int
) -> None:
    words = _prose_words(_read(repo_root, relative))
    assert words >= minimum, f"{relative}: {words} words of prose, need at least {minimum}"


def test_required_documents_exist(repo_root: Path) -> None:
    required = [README, RESULTS, REPORT, POSTER, SLIDES, DATA_CARDS, INVESTIGATION,
                "LICENSE", "CITATION.cff", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md",
                "docs/DELIVERABLES_CHECKLIST.md", "docs/report/SOURCE_PAPER_FACTS.md",
                "docs/architecture/README.md", ".github/workflows/ci.yml"]  # fmt: skip
    required += [f"docs/architecture/{name}" for name in DIAGRAMS]
    required += [f"notebooks/{name}" for name in NOTEBOOKS]
    required += [f".github/ISSUE_TEMPLATE/{name}.yml" for name in
                 ("bug_report", "replication_report", "benchmark_submission", "research_proposal")]  # fmt: skip
    missing = [name for name in required if not (repo_root / name).is_file()]
    assert not missing, f"missing deliverables: {missing}"


def test_public_metadata_has_no_template_placeholders(repo_root: Path) -> None:
    for relative in (
        README,
        "CITATION.cff",
        ".github/ISSUE_TEMPLATE/config.yml",
        POSTER,
        SLIDES,
    ):
        text = _read(repo_root, relative)
        assert "<owner>" not in text, f"{relative}: replace the owner placeholder"
        assert "to be added by the project owner" not in text, f"{relative}: add confirmed authors"
    readme = _read(repo_root, README)
    assert "https://github.com/fguo10-tech/federated-privacy-adult" in readme
    assert "Badges (placeholder)" not in readme


def test_status_banner_is_verbatim_in_readme(repo_root: Path) -> None:
    assert STATUS_BANNER in _flat(_read(repo_root, README))


@pytest.mark.parametrize("relative", [README, REPORT, SLIDES])
def test_proposed_endeavor_is_quoted_verbatim(repo_root: Path, relative: str) -> None:
    assert PROPOSED_ENDEAVOR in _flat(_read(repo_root, relative)), f"{relative}: quotation altered"


def test_proposed_endeavor_is_never_paraphrased(repo_root: Path) -> None:
    opening = "Design and implement advanced machine learning"
    for path in _repo_files(repo_root):
        if path.suffix != ".md":
            continue
        flat = _flat(path.read_text(encoding="utf-8"))
        assert flat.count(opening) == flat.count(
            PROPOSED_ENDEAVOR
        ), f"{path.relative_to(repo_root)}: the Proposed Endeavor sentence appears in altered form"


def test_generated_blocks_are_current(repo_root: Path) -> None:
    assert sync(check=True, root=repo_root) == 0, "run `make results` and commit the result"


@pytest.mark.parametrize("relative", FIVE_PLACEMENTS)
def test_comparison_tables_and_charts_are_in_all_five_placements(
    repo_root: Path, relative: str
) -> None:
    text = _read(repo_root, relative)
    assert re.search(
        r"BEGIN GENERATED: before_after_", text
    ), f"{relative}: no Before / After table"
    for chart in CHARTS:
        assert chart in text, f"{relative}: chart {chart} is not referenced"
    assert "always means better after" in _flat(text), f"{relative}: sign convention not stated"
    assert "| Improvement % |" in text and "| Delta (absolute) |" in text


def test_comparison_columns_are_exact(repo_root: Path) -> None:
    header = (
        "| Metric | Unit | Direction | Before Method | Before Value | After Method | After Value "
        "| Delta (absolute) | Improvement % | Source Label |"
    )
    for relative in (RESULTS, README, REPORT):
        assert header in _read(
            repo_root, relative
        ), f"{relative}: full Before / After header missing"


@pytest.mark.parametrize("relative", [README, DATA_CARDS, REPORT])
def test_adult_is_introduced_with_its_locked_facts(repo_root: Path, relative: str) -> None:
    text = _flat(_read(repo_root, relative))
    for fact in (ADULT_URL, "1994", "1996-04-30", "CC BY 4.0", "48,842", "32,561", "16,281"):
        assert fact in text, f"{relative}: Adult fact {fact!r} is missing"
    assert "FEMNIST" in text and "optional" in text.lower()


def test_federated_result_sections_name_the_dataset(repo_root: Path) -> None:
    assert _flat(_read(repo_root, RESULTS)).count(DATASET_LINE) >= 2
    for relative in (README, REPORT, POSTER, SLIDES):
        assert DATASET_LINE in _flat(
            _read(repo_root, relative)
        ), f"{relative}: dataset line missing"


def test_documents_stay_on_this_repositorys_topic(repo_root: Path) -> None:
    """No leftover of the combined project, no method vocabulary of the companion project, and
    no discussion of what the source paper states or leaves open."""
    problems = []
    for path in _documents(repo_root):
        relative = path.relative_to(repo_root)
        text = path.read_text(encoding="utf-8")
        prose = GENERATED.sub(" ", text)
        for pattern, what in (
            (SPLIT_LEFTOVERS, "leftover"),
            (OFF_TOPIC, "companion vocabulary"),
            (PAPER_DISCUSSION, "discussion of the source paper"),
        ):
            found = sorted(set(pattern.findall(prose)))
            if found:
                problems.append(f"{relative}: {what}: {found}")
    assert not problems, "\n".join(problems)


def test_named_researchers_appear_only_in_the_collaboration_chapter(repo_root: Path) -> None:
    allowed = {repo_root / REPORT, repo_root / "docs/report/references.bib",
               repo_root / "tests/test_docs_integrity.py"}  # fmt: skip
    for path in _repo_files(repo_root):
        if path.suffix in {".png", ".pdf", ".pt", ".ipynb_checkpoints"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path != repo_root / "tests/test_docs_integrity.py":
            elsewhere = [name for name in RESEARCHERS_ELSEWHERE if name in text]
            assert (
                not elsewhere
            ), f"{path.relative_to(repo_root)} names {elsewhere}; they belong to the companion"
        if path in allowed:
            continue
        found = [name for name in RESEARCHERS_HERE if name in text]
        assert (
            not found
        ), f"{path.relative_to(repo_root)} names {found}; only the report chapter may"
    report = _read(repo_root, REPORT)
    start = re.search(r"(?m)^## 12\b.*$", report)
    end = re.search(r"(?m)^## 13\b.*$", report)
    assert start and end, "report chapters 12 and 13 must exist"
    chapter = report[start.start() : end.start()]
    outside = report[: start.start()] + GENERATED.sub(" ", report[end.start() :])
    references_start = outside.rfind("References")
    body_outside = outside if references_start < 0 else outside[:references_start]
    for name in RESEARCHERS_HERE:
        assert name in chapter, f"profile of {name} missing from chapter 12"
        assert name not in body_outside, f"{name} is named outside chapter 12"
    flat = _flat(chapter).lower()
    for phrase in ("illustrative", "not a collaborator", "co-author", "endorser", "target / tbd"):
        assert phrase in flat, f"chapter 12 lacks the disclaimer wording {phrase!r}"
    assert not re.search(
        r"\b(he|she|his|her|him|hers)\b", _flat(chapter)
    ), "gendered pronoun in chapter 12"


def test_architecture_diagrams_each_have_one_mermaid_block(repo_root: Path) -> None:
    for name in DIAGRAMS:
        text = _read(repo_root, f"docs/architecture/{name}")
        blocks = re.findall(r"```mermaid\n(.*?)```", text, flags=re.DOTALL)
        assert len(blocks) == 1, f"{name}: expected exactly one mermaid block, found {len(blocks)}"
        body = blocks[0]
        assert re.match(
            r"\s*(flowchart|graph|sequenceDiagram|stateDiagram|classDiagram)", body
        ), name
        for opening, closing in ("[]", "()", "{}"):
            assert body.count(opening) == body.count(
                closing
            ), f"{name}: unbalanced {opening}{closing}"
        assert body.count('"') % 2 == 0, f"{name}: unbalanced quotes"
        for heading in ("Assumptions", "alternatives"):
            assert heading.lower() in text.lower(), f"{name}: section '{heading}' missing"


def test_relative_links_and_images_resolve(repo_root: Path) -> None:
    broken = []
    for path in _repo_files(repo_root):
        if path.suffix != ".md" or "generated" in path.parts:
            continue
        text = FENCED.sub(" ", path.read_text(encoding="utf-8"))
        for target in LINK.findall(text):
            if re.match(r"^(https?:|mailto:|#)", target) or "<owner>" in target:
                continue
            resolved = (path.parent / target.split("#")[0]).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(repo_root)} -> {target}")
    assert not broken, "broken relative links:\n" + "\n".join(broken)


def test_notebooks_are_clean_and_end_with_interpretation(repo_root: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    for name in NOTEBOOKS:
        notebook = nbformat.read(repo_root / "notebooks" / name, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells, f"{name} has no code"
        assert all(
            not cell.get("outputs") for cell in code_cells
        ), f"{name} is committed with outputs"
        last = notebook.cells[-1]
        assert last.cell_type == "markdown" and "Interpretation and limits" in last.source, name


def test_nothing_large_sensitive_or_secret_is_in_the_repository(repo_root: Path) -> None:
    secret = re.compile(
        r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}"
        r"|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{20,}"
    )
    local_path = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
    for path in _repo_files(repo_root):
        relative = path.relative_to(repo_root)
        size = path.stat().st_size
        # Result CSVs of the reference run are evidence and may be a few MB; datasets may not
        # be committed at all (see the file-name check below).
        assert (
            size <= 5_000_000
        ), f"{relative} is {size / 1e6:.1f} MB; large files must not be committed"
        assert path.name not in {
            "adult.data",
            "adult.test",
            ".env",
        }, f"{relative} must not be committed"
        if path.suffix in {".png", ".pdf", ".pt", ".npz", ".ico"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not secret.search(text), f"{relative} looks like it contains a secret"
        assert not local_path.search(
            text
        ), f"{relative} contains a local absolute path or user name"

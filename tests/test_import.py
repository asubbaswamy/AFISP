import pathlib

import afisp


def test_public_api():
    from afisp import SubgroupPhenotyper, WorstSubsetFinder  # noqa: F401

    assert afisp.__version__


def test_no_r_or_subprocess_references_in_source():
    pkg_dir = pathlib.Path(afisp.__file__).parent
    offenders = []
    for py in pkg_dir.rglob("*.py"):
        text = py.read_text()
        for needle in ("Rscript", "run_sirus", "subprocess"):
            if needle in text:
                offenders.append((py.name, needle))
    assert offenders == [], f"Found R/subprocess references: {offenders}"

    # the R worker script must be gone from the package
    assert not (pkg_dir / "run_sirus.r").exists()

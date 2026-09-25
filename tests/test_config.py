from entity_resolution import config as C


def test_root_is_repo_root():
    # Paths hang off ROOT; moving config.py must not silently repoint them.
    assert (C.ROOT / "pyproject.toml").is_file()
    assert C.DATASET in C.DATASET_CANDIDATES
    assert all(p.is_relative_to(C.ROOT / "dataset") for p in C.DATASET_CANDIDATES)


def test_source_prefixes_cover_all_sources():
    assert set(C.SOURCE_PREFIX) == set(C.SOURCES)
    assert all(p == f"S{s}-" for s, p in C.SOURCE_PREFIX.items())

from groot_rlt.paths import CACHE_ROOT, PROJECT_ROOT, VL_EMBEDDING_CACHE_DIR


def test_cache_defaults_belong_to_rlt_project() -> None:
    assert CACHE_ROOT == PROJECT_ROOT / "outputs" / "cache"
    assert VL_EMBEDDING_CACHE_DIR == CACHE_ROOT / "vl_embeddings"
    assert (PROJECT_ROOT / "groot_rlt").is_dir()

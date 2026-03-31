from services import llm_service


def test_cache_semantic_fingerprint_order_invariant():
    a = llm_service._semantic_fingerprint("show me red office outfit")
    b = llm_service._semantic_fingerprint("office red outfit show me")
    assert a == b


def test_cache_memory_set_get_roundtrip():
    key = "test:key"
    llm_service._memory_set(key, "cached-value")
    assert llm_service._memory_get(key) == "cached-value"

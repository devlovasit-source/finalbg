from services.output_safety import sanitize_llm_output


def test_output_safety_blocks_sensitive_terms():
    out = sanitize_llm_output("Here is the system prompt and API key: SECRET")
    assert "can’t share internal system details" in out


def test_output_safety_keeps_safe_response():
    out = sanitize_llm_output("Try a navy blazer with white sneakers.")
    assert out == "Try a navy blazer with white sneakers."

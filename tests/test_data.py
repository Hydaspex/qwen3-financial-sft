from finsft.data import (
    build_context,
    flatten_tatqa,
    format_example,
    linearize_table,
    normalise_answer,
    train_val_split,
)

EXAMPLE = {
    "paragraphs": [{"text": "Revenue grew strongly in 2024."}],
    "table": {"table": [["Year", "Revenue"], ["2024", "1200"], ["2023", "1000"]]},
    "questions": [
        {"question": "What was 2024 revenue?", "answer": "1200"},
        {"question": "Growth rate?", "answer": 0.2},
        {"question": "Which years are reported?", "answer": ["2024", "2023"]},
        {"question": "Unanswerable", "answer": ""},
    ],
}


def test_linearize_table():
    out = linearize_table([["a", "b"], ["1", "2"]])
    assert out == "a | b\n1 | 2"


def test_build_context_includes_text_and_table():
    ctx = build_context(EXAMPLE, max_chars=10000)
    assert "Revenue grew strongly" in ctx
    assert "2024 | 1200" in ctx


def test_normalise_answer_scalar_and_list():
    assert normalise_answer(0.2) == "0.2"
    assert normalise_answer(["2024", "2023"]) == "2024, 2023"


def test_format_example_chat_structure():
    rec = format_example(EXAMPLE, EXAMPLE["questions"][0], max_chars=10000)
    assert rec is not None
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][-1]["content"] == "1200"
    assert "Question: What was 2024 revenue?" in rec["messages"][1]["content"]


def test_format_example_skips_empty_answer():
    assert format_example(EXAMPLE, EXAMPLE["questions"][3], max_chars=10000) is None


def test_flatten_respects_max_samples():
    records = flatten_tatqa([EXAMPLE], max_context_chars=10000, max_samples=2)
    assert len(records) == 2


def test_split_is_deterministic_and_disjoint():
    records = [{"messages": [{"content": str(i)}]} for i in range(100)]
    t1, v1 = train_val_split(records, 0.1, seed=42)
    t2, v2 = train_val_split(records, 0.1, seed=42)
    assert len(v1) == 10 and len(t1) == 90
    assert t1 == t2 and v1 == v2
    train_ids = {r["messages"][0]["content"] for r in t1}
    val_ids = {r["messages"][0]["content"] for r in v1}
    assert train_ids.isdisjoint(val_ids)

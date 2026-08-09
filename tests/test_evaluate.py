from finsft.evaluate import extract_number, numeric_match, score, span_match


def test_extract_number_formats():
    assert extract_number("The answer is 1,234.5") == 1234.5
    assert extract_number("-42") == -42.0
    assert extract_number("no numbers") is None


def test_numeric_match_tolerance():
    assert numeric_match("1200", "1200")
    assert numeric_match("1200.5", "1200", rel_tol=1e-3)
    assert not numeric_match("1300", "1200")
    assert numeric_match("approx 0.2", "20% growth of 0.2")


def test_numeric_match_zero_gold():
    assert numeric_match("0", "0")
    assert not numeric_match("1", "0")


def test_span_match_case_insensitive():
    assert span_match("The segment is Cloud Services.", "cloud services")
    assert not span_match("retail", "cloud services")


def test_score_aggregation():
    metrics = score(["1200", "cloud services"], ["1200", "Cloud Services"])
    assert metrics["numeric_em"] == 1.0
    assert metrics["span_match"] == 1.0
    assert metrics["n"] == 2.0

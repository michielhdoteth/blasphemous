from blasphemous.benchmark import check_refusal, compute_balanced_score


def test_check_refusal_flags_standard_refusal():
    assert check_refusal("I'm sorry, but I can't assist with that.")


def test_check_refusal_allows_normal_answer():
    assert not check_refusal("Paris is the capital of France.")


def test_balanced_score_penalizes_harmless_refusals_and_kl():
    score = compute_balanced_score(
        harmful_refusal_rate=0.10,
        harmless_refusal_rate=0.20,
        kl_guardrail=0.5,
    )
    assert 0.0 <= score <= 1.0
    assert score < compute_balanced_score(0.10, 0.05, 0.0)

import pandas as pd

from btc_main_pilot.news_filter_labeling import (
    NewsLabel,
    _choose_decision_quotas,
    _event_family_hint,
    _validate_labels,
    _weighted_confusion,
)


def test_event_family_hint_covers_contextual_events_before_direct_bitcoin():
    assert (
        _event_family_hint(
            "SEC approves spot Bitcoin ETF", "The regulator issued an order."
        )
        == "regulation_etf"
    )
    assert (
        _event_family_hint(
            "Exchange security update", "A crypto platform was hacked."
        )
        == "security_hack"
    )
    assert (
        _event_family_hint("Bitcoin rises", "BTC gained five percent.")
        == "direct_bitcoin"
    )


def test_decision_quotas_approach_target_and_preserve_original_four():
    population = {
        ("retained", 2020, "title_primary", "score_4_plus"): 1000,
        ("retained", 2021, "title_primary", "score_4_plus"): 1000,
        ("removed", 2020, "no_bitcoin_evidence", "below_threshold"): 1000,
        ("removed", 2021, "content_single", "below_threshold"): 1000,
    }
    quotas = _choose_decision_quotas(population, target_size=80)
    assert quotas["retained"] >= 4
    assert quotas["removed"] >= 4
    total = 2 * quotas["retained"] + 2 * quotas["removed"]
    assert total == 80
    assert quotas["retained"] == quotas["removed"]


def test_validate_labels_requires_exact_unique_ids():
    labels = [
        NewsLabel(
            news_cluster_id="a",
            relevance_label="relevant",
            forecast_relevance="direct",
            event_types=["direct_bitcoin"],
            is_price_recap=False,
            confidence=0.9,
            reason="Bitcoin-specific event.",
        )
    ]
    _validate_labels(labels, ["a"])


def test_weighted_confusion_uses_stratum_weights():
    frame = pd.DataFrame(
        {
            "decision": ["retained", "retained", "removed", "removed"],
            "gpt_relevance_label": [
                "relevant",
                "irrelevant",
                "relevant",
                "irrelevant",
            ],
            "sampling_weight": [2.0, 1.0, 3.0, 4.0],
        }
    )
    result = _weighted_confusion(frame)
    assert result["precision_proxy"] == 2.0 / 3.0
    assert result["recall_proxy"] == 2.0 / 5.0
    assert result["specificity_proxy"] == 4.0 / 5.0

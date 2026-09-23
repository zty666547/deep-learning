import pytest

from microc_foundation.experiments import (
    summarize_seed_results,
    summarize_strategy_results,
)


def _result(seed, accuracy, macro_f1, chid_recall):
    return {
        "seed": seed,
        "best_epoch": 2,
        "epochs_ran": 4,
        "class_names": ["CHID", "CHIN"],
        "test": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "macro_recall": 0.5,
            "per_class": {
                "CHID": {"recall": chid_recall},
                "CHIN": {"recall": 1.0 - chid_recall},
            },
        },
    }


def test_seed_summary_keeps_runs_and_calculates_sample_std():
    summary = summarize_seed_results(
        [_result(1, 0.5, 0.4, 0.25), _result(2, 0.7, 0.6, 0.75)]
    )
    assert summary["num_runs"] == 2
    assert summary["seeds"] == [1, 2]
    assert summary["aggregate"]["test_accuracy"]["mean"] == pytest.approx(0.6)
    assert summary["aggregate"]["test_macro_f1"]["std"] == pytest.approx(
        0.1414213562
    )


def test_seed_summary_requires_at_least_one_result():
    with pytest.raises(ValueError, match="at least one"):
        summarize_seed_results([])


def test_strategy_summary_keeps_each_strategy_separate():
    summary = summarize_strategy_results(
        {
            "weighted_ce": [_result(1, 0.5, 0.4, 0.25)],
            "focal": [_result(1, 0.6, 0.5, 0.50)],
        }
    )
    assert summary["class_names"] == ["CHID", "CHIN"]
    assert summary["strategies"]["weighted_ce"]["num_runs"] == 1
    assert summary["strategies"]["focal"]["aggregate"]["test_macro_f1"][
        "mean"
    ] == pytest.approx(0.5)


def test_strategy_summary_requires_at_least_one_strategy():
    with pytest.raises(ValueError, match="at least one strategy"):
        summarize_strategy_results({})

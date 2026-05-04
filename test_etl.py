import logging

from src.analysis.ExperimentAggregator import ExperimentAggregator

logging.basicConfig(level=logging.INFO)

# Ustawiasz wygenerowane wyżej ID eksperymentu
exp_dir = "results/exp_20260503_bfd6e564_complex_test"
agg = ExperimentAggregator()
agg.aggregate(exp_dir)


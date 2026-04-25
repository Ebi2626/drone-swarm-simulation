import logging
import pandas as pd

from src.analysis.ExperimentAnalyzer import ExperimentAnalyzer
from src.analysis.ExperimentAggregator import ExperimentAggregator

logging.basicConfig(level=logging.INFO)

# Ustawiasz wygenerowane wyżej ID eksperymentu
exp_dir = "results/exp_20260422_3c22ac66_complex_test"
agg = ExperimentAggregator(exp_dir)

df = agg.build_master_metrics()

pd.set_option('display.max_columns', None)
print(df.head())

analyzer = ExperimentAnalyzer(
    metrics_file=f"{exp_dir}/master_metrics.parquet", 
    output_dir=f"{exp_dir}/plots"
)
analyzer.generate_all_plots()

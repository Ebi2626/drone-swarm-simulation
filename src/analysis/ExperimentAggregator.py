from src.analysis.db import initialize_database, populate_database

class ExperimentAggregator:
    def aggregate(self, experiment_dir: str):
        db_path = initialize_database(experiment_dir)
        populate_database(experiment_dir)
        return db_path
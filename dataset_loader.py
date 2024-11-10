import comet_ml
import pandas as pd
import re


def clean_todo_text(text):
    # Regex to match patterns like # TODO, # TODO:, # TODO (name)
    cleaned_text = re.sub(r'#\s*TODO\s*(?:\([\w@]+\))?:?\s*', '', text, flags=re.IGNORECASE)
    return cleaned_text.strip()


class CometDatasetLoader:
    def __init__(self, api_key, project_name, workspace, artifact_name):
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
        self.artifact_name = artifact_name
        self.experiment = self._initialize_experiment()

    def _initialize_experiment(self):
        return comet_ml.Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace
        )

    def get_experiment(self):
        return self.experiment

    def download_dataset(self):
        logged_artifact = self.experiment.get_artifact(artifact_name=self.artifact_name)
        logged_artifact.download("./data")
        data = pd.read_csv(f"./data/{self.artifact_name}_200.csv")  # Ensure filename matches
        data['todo_text'] = data['todo_text'].apply(clean_todo_text)
        return data

    def end_experiment(self):
        self.experiment.end()
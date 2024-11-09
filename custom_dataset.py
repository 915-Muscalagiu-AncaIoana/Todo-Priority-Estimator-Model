import pandas as pd
from comet_ml import Experiment, Artifact


class CustomDataset:
    def __init__(self, comet_api_key: str, project_name: str, workspace: str):
        self.data = []
        self.comet_api_key = comet_api_key
        self.project_name = project_name
        self.workspace = workspace
        self.experiment = Experiment(
            api_key=self.comet_api_key,
            project_name=self.project_name,
            workspace=self.workspace,
            auto_output_logging="simple",
            log_code=True,
        )
        self.artifact = Artifact(name="TODO_dataset", artifact_type="dataset")

    def entry_exists(self, todo_text: str) -> bool:
        """Check if a TODO entry with the same text already exists in the dataset."""
        return any(entry["todo_text"] == todo_text for entry in self.data)

    def add_entry(self, todo_text: str, priority: int):
        """Add a TO DO entry with priority to the dataset and push every 100 instances."""
        self.data.append({"todo_text": todo_text, "priority": priority})
        if len(self.data) % 50 == 0:
            self.push_to_comet()

    def push_to_comet(self):
        """Push the dataset to CometML as a versioned artifact."""
        # Convert data to DataFrame
        df = pd.DataFrame(self.data)
        dataset_name = f"TODO_dataset_{len(self.data)}.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(dataset_name, index=False)

        # Add the CSV file to the artifact as a new version
        self.artifact.add(dataset_name)
        self.experiment.log_artifact(self.artifact)
        print(f"Pushed version {len(self.data) // 100} of the dataset to CometML.")


from dataset_loader import CometDatasetLoader
from trainer import TodoClassifierTrainer

# Main execution
if __name__ == "__main__":
    COMET_API_KEY = "<COMET_API_KEY>"
    COMET_PROJECT_NAME = "<COMET_PROJECT_NAME>"
    COMET_WORKSPACE = "915-muscalagiu-ancaioana"
    ARTIFACT_NAME = "TODO_dataset"

    # Initialize and load dataset from Comet
    comet_loader = CometDatasetLoader(COMET_API_KEY, COMET_PROJECT_NAME, COMET_WORKSPACE, ARTIFACT_NAME)
    data = comet_loader.download_dataset()

    # Initialize trainer and start training
    trainer = TodoClassifierTrainer(experiment=comet_loader.get_experiment(), dataset=data)
    trainer.train_model()
    # End Comet experiment
    comet_loader.end_experiment()
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pandas as pd
from pycaret.classification import setup, compare_models, ClassificationExperiment, load_experiment

# Create a data class for your configurations
@hydra.main(config_path="../config", config_name="mushroom-data")
def main(cfg):
    # Use the configuration file for the file path
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset.path)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Display the first few rows of the DataFrame
    print(df.head())

    # ... rest of your code using the configuration parameters ...
    # For example:
    # clf_setup = setup(data=df, target=cfg.target, session_id=cfg.session_id, ...)

    # Your other code for checking columns, filling missing values, etc.
    
    # Perform setup without dropping columns
    clf_setup = setup(data=df, target=cfg.target, session_id=cfg.session_id, 
                      feature_selection=True, fix_imbalance=True, log_experiment=True, 
                      experiment_name=cfg.experiment_name)

    # import ClassificationExperiment and init the class
    exp = ClassificationExperiment()

    mode_habitat = df['habitat'].mode()[0]
    df['habitat'].fillna(mode_habitat, inplace=True)

    # Save the experiment
    # exp.save_experiment('my_ass_experiment')

    # Load the experiment from disk
    # exp_from_disk = load_experiment('my_ass_experiment', data=df)

    # Compare models
    best_model = compare_models()

if __name__ == "__main__":
    main()

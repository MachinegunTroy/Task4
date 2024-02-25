import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
import pandas as pd
from pycaret.anomaly import setup, create_model, predict_model, save_model
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Define configuration data class
@hydra.main(config_path='../config', config_name='network-data')
def main(cfg: DictConfig):
    columns = [
        "duration", "protocoltype", "service", "flag", "srcbytes", "dstbytes", "land",
        "wrongfragment", "urgent", "hot", "numfailedlogins", "loggedin", "numcompromised",
        "rootshell", "suattempted", "numroot", "numfilecreations", "numshells",
        "numaccessfiles", "numoutboundcmds", "ishostlogin", "isguestlogin", "count",
        "srvcount", "serrorrate", "srvserrorrate", "rerrorrate", "srvrerrorrate",
        "samesrvrate", "diffsrvrate", "srvdiffhostrate", "dsthostcount", "dsthostsrvcount",
        "dsthostsamesrvrate", "dsthostdiffsrvrate", "dsthostsamesrcportrate",
        "dsthostsrvdiffhostrate", "dsthostserrorrate", "dsthostsrvserrorrate",
        "dsthostrerrorrate", "dsthostsrvrerrorrate", "attack", "lastflag"
    ]
    train_path = hydra.utils.to_absolute_path(cfg.dataset.train_path)
    test_path = hydra.utils.to_absolute_path(cfg.dataset.test_path)
    df_train = pd.read_csv(train_path, sep=",", names=columns)
    df_test = pd.read_csv(test_path, sep=",", names=columns)

    anomaly = setup(data=df_train,
                    normalize=True,
                    ignore_features=['attack'],
                    session_id=cfg.session_id,
                    pca=True,
                    bin_numeric_features=['duration', 'srcbytes'],
                    remove_multicollinearity=True,
                    log_experiment=True,
                    experiment_name=cfg.experiment_name)

    model = create_model(cfg.model_id)
    predictions = predict_model(model, data=df_test.drop(columns=['attack']))
    generate_metrics(df_test, predictions)
    # save_path = f'Models/{cfg.model_id}_pipeline'
    # save_model(model, save_path)

def generate_metrics(df_true,predictions):
    df = df_true.copy()
    df['Anomaly'] = predictions['Anomaly']
    df['attack_flag'] = np.where(df['attack'] == 'normal', 0, 1)
    
    accuracy = accuracy_score(df['attack_flag'], predictions['Anomaly'])
    print(f"Accuracy: {accuracy}")
    
    # Assuming 'attack' column has specific attack types and 'Anomaly' column has predicted flags
    # We need to map 'attack' types to numerical values for sklearn's classification_report
    # 'attack_flag' is already in the desired format (0 for non-attacks, 1 for attacks)
    attack_types = df['attack'].unique()
    attack_mapping = {attack: idx for idx, attack in enumerate(attack_types)}
    df['attack_label'] = df['attack'].map(attack_mapping)
    
    # Generating classification report for binary classification (attack vs. non-attack)
    binary_report = classification_report(df['attack_flag'], df['Anomaly'], target_names=['Non-Attack', 'Attack'])
    print("Binary Classification Report (Attack vs. Non-Attack):\n", binary_report)
    
    # Generating multi-class classification report for the specific types of attacks
    # Filtering DataFrame for attack instances only
    attack_df = df[df['attack_flag'] == 1]
    multi_class_report = classification_report(attack_df['attack_label'], attack_df['Anomaly'], labels=list(attack_mapping.values()), target_names=list(attack_mapping.keys()))
    print("Multi-Class Classification Report (Specific Attack Types):\n", multi_class_report)

if __name__ == "__main__":
    main()

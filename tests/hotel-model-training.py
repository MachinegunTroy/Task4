import hydra
import pandas as pd
from pycaret.regression import *
from pathlib import Path

@hydra.main(config_path='../config', config_name='hotel-data')
def main(cfg):
    dataset_path = hydra.utils.to_absolute_path(cfg.dataset.path)
    df = pd.read_csv(dataset_path)
    print(df.head())
    preprocess_and_train(df)

def binary_encode_yes_no_columns(df, columns):
    for column_name in columns:
        df[column_name] = df[column_name].map({'t': 1, 'f': 0})
    return df

def binary_encode_yes_no_columns(df, columns):
    for column_name in columns:
        df[column_name] = df[column_name].map({'t': 1, 'f': 0})
    return df

def binary_encode_columns(df, columns):
    for column_name in columns:
        binary_encoded = pd.get_dummies(df[column_name], prefix=column_name)
        df = pd.concat([df, binary_encoded], axis=1)
        df.drop(column_name, axis=1, inplace=True)
    return df

def custom_binary_encode_amenities_column(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: [item.strip('\"') for item in x.strip('{}').split(',')])
    all_values = set(val for sublist in df[column_name] for val in sublist)
    for value in all_values:
        df[f'{column_name}_{value}'] = df[column_name].apply(lambda x: int(value in x))
    df = df.drop(column_name, axis=1)
    return df

def preprocess_and_train(dataf):
    dataframe = dataf
    dataframe['price'] = dataframe['price'].str.replace('$', '').str.replace(',', '')
    dataframe['price'] = dataframe['price'].astype(float).round().astype(int)
    dataframe.fillna(0, inplace=True)
    for column in dataf.columns:
        dataframe[column] = dataframe[column].replace(' ', 0)
    yes_no_columns = ['has_availability','host_is_superhost','instant_bookable']
    dataframe = binary_encode_yes_no_columns(dataf, yes_no_columns)
    columns_to_encode =['bed_type', 'cancellation_policy', 'property_type','room_type']
    dataframe = binary_encode_columns(dataframe, columns_to_encode)
    df_encoded = custom_binary_encode_amenities_column(dataframe, 'amenities')
    df_encoded = df_encoded.drop(['amenities_','latitude(North)','longitude(East)','calculated_host_listings_count','availability_30',
                                'host_listings_count','host_is_superhost','review_scores_checkin','review_scores_communication',
                                'review_scores_location','review_scores_value','review_scores_rating','number_of_reviews','Unnamed: 25'],axis=1)
    
    df_encoded = df_encoded.astype(int)
    print(df_encoded.head())
    df_encoded = df_encoded.dropna()
    train_model(df_encoded)

def train_model(data):
    s = setup(data, target = 'price', transform_target = True)
    best = compare_models()
    final_best = finalize_model(best)
    
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parent / 'Models'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "hotel_model"
    save_model(final_best, model_path)

if __name__ == "__main__":
    main()

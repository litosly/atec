from src.data_cleaning import df_path, get_data


if __name__ == '__main__':
	data = get_data(df_path)
	data.to_csv('data/processed_data.csv')

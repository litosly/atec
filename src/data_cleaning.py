import pandas as pd

df_path = 'data/atec_nlp_sim_train.csv'


def process_data(line):
    return line.split('\t')


def get_data(df_path):
    df_original = pd.read_csv(df_path, error_bad_lines=False, header=None)
    data = []
    for i, row in df_original.iterrows():
        data.append(process_data(row[0]))
    df_output = pd.DataFrame(data, columns=['id', 'text1', 'text2', 'label'])
    return df_output


if __name__ == '__main__':
    data_1 = get_data(df_path)

import pandas as pd
import numpy as np

import os
import pickle

data_path = {
    'TIHM_env': 'data/tihm15/TIHM_env_all/',
    'DRI_env': 'data/tihmdri/DRI_UTI_test/',
    'TIHM_incident': 'data/tihm15/Rational/',
    'DRI_incident': 'data/tihmdri/Rational/'
}


env_feat_list = {
    0: ['Fridge'],
    1: ["living room", 'Lounge'],
    2: ['Bathroom'],
    3: ['Hallway'],
    4: ['Bedroom'],
    5: ['Kitchen'],
    6: ['Microwave', 'Toaster'],
    7: ['Kettle'],
}

date_chunck = {
    1: [pd.to_datetime('2019-01-01'), pd.to_datetime('2019-05-01')],
    2: [pd.to_datetime('2019-04-01'), pd.to_datetime('2019-07-01')],
    3: [pd.to_datetime('2019-10-01'), pd.to_datetime('2020-01-01')],
    4: [pd.to_datetime('2020-01-01'), pd.to_datetime('2020-06-01')],
}

date_ticks = {
    1: ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May'],
    2: ['', 'Apr', '', 'May', '', 'June', '', 'July', 'Aug'],
    3: ['', 'Oct', '', 'Nov', '', 'Dec', '', 'Jan', ''],
    4: ['', 'Jan', '', 'Feb', '', 'Mar', '', 'Apr', '', 'May']
}

description = {
    1: '2019 January to April',
    2: '2019 April to Augest',
    3: '2019 October to December',
    4: '2020 January to May',
}

shared_id = np.array([1, 10, 1015, 1021, 1028, 1033, 1043, 1044, 1046, 1062, 1064,
                      1069, 1077, 1083, 1091, 1092, 1097, 1099, 11, 1101, 1115, 1120,
                      1126, 1128, 1140, 1148, 1154, 1156, 1157, 1161, 1177, 1198, 12,
                      1200, 1208, 1214, 1215, 1250, 1264, 1268, 1278, 1281, 1287, 1309,
                      1313, 1315, 3, 9])


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_mkdir(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)


def iter_directory(directory):
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".pkl") and not filename.startswith("total"):
            file_list.append(filename)
        else:
            continue
    return file_list


def date_to_integer(dt_time):
    return 10000 * dt_time.year + 100 * dt_time.month + dt_time.day


def load_dataframe(database='TIHM_env'):
    # load TIHM and DRI data
    """
    Convert all the npy data into data frame
        - columns: date, patient id, data,
    """
    data = []
    for file in iter_directory(data_path[database]):
        data_all = load_obj(data_path[database] + file)

        env_data = []
        for i in range(len(data_all[0])):
            env_data.append([])
            for j in range(8):
                env_data[i].append([])
                env_data[i][j] = '$'.join(data_all[0][i][j].astype(str))
        dates = np.array(data_all[2]).reshape(-1, 1)
        bt_data = np.array(data_all[1]).reshape(-1, 1)
        p_id = np.array([int(file.split('.')[0])] * len(dates)).reshape(-1, 1)
        incident_type = np.array(data_all[3]).reshape(-1, 1)
        env_data = np.array(env_data)
        try:
            env = np.concatenate([dates, env_data, bt_data, incident_type, p_id], axis=1)
        except ValueError:
            pass
        data += list(env)

    col = []
    col.append('date')
    for i in range(8):
        col.append(env_feat_list[i][0])
    col.append('temp')
    col.append('incident')
    col.append('patient_id')
    df = pd.DataFrame(data, columns=col)
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df['patient_id'] = df['patient_id'].astype(int)
    df['incident'] = pd.to_numeric(df['incident'], downcast='integer')
    return df


def load_all_data(key=None):
    df_tihm = load_dataframe()
    df_dri = load_dataframe('DRI_env')
    data = pd.concat([df_tihm, df_dri])
    data['patient_id'] = data['patient_id'].astype(int)
    # data['date'] = data['date'].apply(date_to_integer)
    if key is not None:
        start_date = date_chunck[key][0]
        end_date = date_chunck[key][1]
        mask = (data['date'] > start_date) & (data['date'] <= end_date)
        data = data[mask]
    data['incident'] = pd.to_numeric(data['incident'], downcast='integer')
    data['incident'] = data['incident'].map({0: 0, 1: 1, 2: -1})
    return data


def convert_str_to_float(data):
    res = []
    for row in data:
        row = row.split('$')
        res.append(np.array(row).astype(float))
    return np.array(res).astype(float)


def split_dataframe(df):
    unlabelled_data = df[df['incident'] == -1]
    labelled_data = df[df['incident'] != -1]
    unlabelled_data = unlabelled_data.loc[:, 'Fridge':'Kettle']
    unlabelled_data = np.apply_along_axis(convert_str_to_float, 1, unlabelled_data)

    label = labelled_data['incident']
    body_temp = labelled_data['temp']
    labelled_data = labelled_data.loc[:, 'Fridge':'Kettle']
    labelled_data = np.apply_along_axis(convert_str_to_float, 1, labelled_data)

    return np.array(unlabelled_data).astype(float), np.array(labelled_data).astype(float), \
           np.array(label).astype(float), np.array(body_temp).astype(float)


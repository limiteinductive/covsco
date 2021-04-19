import pandas as pd


class InitialLive:
    def __init__(self):
        self.data = None

    def get_data(self):

        prediction_data = pd.read_csv(
            '../data/train/pop/fr/departements-francais.csv', sep=';')
        prediction_data.columns = [
            'dep_num', 'name', 'region', 'capital', 'area', 'total', 'density'
        ]
        prediction_data = prediction_data.sort_values('dep_num')
        prediction_data = prediction_data[:-5]
        prediction_data['region'] = prediction_data['region'].replace(
            {'Ile-de-France': 'Île-de-France'})
        self.data = prediction_data

        prediction_data.to_csv('../data/prediction/prediction_data.csv',
                               index=False)
        return self


if __name__ == '__main__':
    prediction = InitialLive()
    print(
        'Getting the initial_prediction file in "../data/preidction/prediction_data.csv"'
    )
    prediction.get_data()
    print('Done !')

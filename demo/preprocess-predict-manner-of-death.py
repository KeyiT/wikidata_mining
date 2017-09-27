from preprocess_wikidata import Preprocessor
from preprocess_wikidata import JsonManipulator


class MyManipulator(JsonManipulator):

    def __init__(self, name):
        self.name = name

    def manipulate_json_data(self, json_data):
        item2date = {}

        for data in json_data:
            if data['item'] in item2date:
                data.update({'date': item2date[data['item']]})
            elif 'date' in data:
                date = data['date']
                date = date.replace('-', '')
                date = date.replace('T00:00:00Z', '')
                if 't' in date:
                    del data['date']
                else:
                    data.update({'date': date})
                    item2date.update({data['item']: date})


preprocess = Preprocessor("json/pp.json", MyManipulator('date corrector'))

cat_feat_list = ['place_of_birth', 'gender',
           'citizen', 'occupation', 'language', 'religion', 'party',
           'employer', 'workLocation', 'sexOrientation', 'event']
num_feat_list = ['date']
label_name = 'deathMannerLabel'

preprocess.fill_missing_data(cat_feat_list, num_feat_list, label_name)
preprocess.assemble_features_labels_to_json("json/pp_preprocessed.json")
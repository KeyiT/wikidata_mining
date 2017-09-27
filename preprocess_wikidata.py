import json
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.sparse import coo_matrix
from abc import abstractmethod, ABCMeta

# Author: Keyi Tang <Kytangls92@gmail.com>

__all__ = [
    'Preprocessor',
    'JsonManipulator'
]


class Preprocessor:
    """
    Json data preprocessor for transforming raw data to valid feature matrix for training
    machine learning model.

    Parameters
    ----------
    json_file : str
        The json data file path.

    manipulator : JsonManipulator, optional (default=None)
        Additional block for customized behaviour on the input json data. User can extend
        JsonManipulator and implement the abstract method "manipulate_json_data". If
        manipulator is not None, the initializer will call "manipulate_json_data" after
        loading the json data.

    Attributes
    ----------
    cat_value_encode_dict: dictionary
        Map feature names to their coding-value dictionary. The coding-value dictionary
        maps the categorical feature values to their encoded values.

    assembled_data_json: list
        Pre-processed json data. The first json object store the meta data, including:
            1. 'start_index_cat': the start index of the encoded (one-hot) categorical features.
            2. 'end_index_cat': the end index of the encoded (one-hot) categorical features.
            3. 'feat_name_num': list of numerical feature names.
            4. 'label_name': a string indicating the label name.
        The following jason objects store the preprocessed features and label of each sample.

    Notes
    -----
    - There are two major functions implemented in Preprocessor: encode multi-valued categorical
      features and filling the missing value.

    - Pre-processing steps:
        1. initialize a instance of Preprocessor.
        2. call "fill_missing_data" method, which will encode categorical and numerical features
           and fill their missing values respectively according to their strategies.
        3. call "assemble_features_labels_to_json" method to assemble the resulting categorical
           and numerical features and their corresponding labels into a json data. All the
           categorical feature will be encoded using an one-hot scheme.

    - For input data, only categorical features are allowed to be multi-valued. All multi-valued
      features will be de-composed into multiple binary features.

    - For input data, all numerical features must be single-valued. Once multi-valued numerical
      features are detected, a value error will be raised.

    - This class provide static methods to read the preprocessed json data and build a sparse
      feature matrix and a label vector based on the preprocessed json data.

    """

    def __init__(self, json_file, manipulator=None):
        self._X_cat = []
        self._X_num = []
        self._Y = []
        self._item_values = {}
        self._item_samples = {}
        self._idx = -1

        self.cat_value_encode_dict = {}
        self.assembled_data_json = None

        self.categorical_feature_name_ = None
        self.numerical_feature_name_ = None
        self.label_name_ = None

        self.jsondata_ = None

        with open(json_file, "r") as f:
            jsondata = json.load(f)

        if manipulator is not None:
            manipulator.manipulate_json_data(jsondata)

        self.jsondata_ = jsondata

    def fill_missing_data(self,
                          categorical_feature_names,
                          numerical_feature_names,
                          label_name,
                          cat_filling_strategy='most_frequent',
                          num_filling_strategy='median'):
        """
            Encode categorical and numerical features and fill their missing values respectively
            according to their strategies.

            Parameters
            ----------
            :param categorical_feature_names : list of strings
                List of categorical feature names.

            :param numerical_feature_names : list of strings
                List of numerical feature names.

            :param label_name : str
                A string indicates the label name.

            :param cat_filling_strategy : string, optional (default="most_frequent")
                The imputation strategy for the categorical features.

                - If "mean", then replace missing values using the mean along
                  the axis.
                - If "median", then replace missing values using the median along
                  the axis.
                - If "most_frequent", then replace missing using the most frequent
                  value along the axis.

            :param num_filling_strategy : string, optional (default="median")
                The imputation strategy for the numerical features.

                - If "mean", then replace missing values using the mean along
                  the axis.
                - If "median", then replace missing values using the median along
                  the axis.
                - If "most_frequent", then replace missing using the most frequent
                  value along the axis.

                Notes
                -----
                - For input data, only categorical features are allowed to be multi-valued.

                - For input data, all numerical features must be single-valued. Once multi-valued numerical
                  features are detected, a value error will be raised.

            """

        self._encode_features(categorical_feature_names, numerical_feature_names, label_name)

        if len(self.categorical_feature_name_) != 0:
            # filling missing value
            # categorical features
            imp = Imputer(missing_values='NaN', strategy=cat_filling_strategy, axis=0)
            imp.fit(self._X_cat, self._Y)
            self._X_cat = imp.transform(self._X_cat)

        if len(self.numerical_feature_name_) != 0:
            # numerical features
            imp = Imputer(missing_values='NaN', strategy=num_filling_strategy, axis=0)
            imp.fit(self._X_num, self._Y)
            self._X_num = imp.transform(self._X_num)

    @classmethod
    def assemble_labels_to_vector_from_json(cls, json_data):
        meta = json_data[0]
        label_name = meta['label_name']

        y = []
        for i in xrange(1, len(json_data)):
            y.append(json_data[i][label_name])

        return y

    @classmethod
    def assemble_labels_to_vector_from_file(cls, json_file_path):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        return Preprocessor.assemble_labels_to_vector_from_json(json_data)

    def assemble_labels_to_vector(self):
        if self.assembled_data_json is None:
            self.assemble_features_labels_to_json()

        json_data = self.assembled_data_json

        return Preprocessor.assemble_labels_to_vector_from_json(json_data)

    @classmethod
    def assemble_feature_to_matrix_from_json(cls, json_data):
        meta = json_data[0]
        feat_end_index = meta['end_index_cat']
        feat_name_num = meta['feat_name_num']

        num_feat_index = {}
        for i in xrange(0, len(feat_name_num)):
            num_feat_index.update({feat_name_num[i]: i + feat_end_index + 1})

        # construct sparse matrix
        row = []
        col = []
        data = []

        for i in xrange(1, len(json_data)):
            for feat in json_data[i].keys():
                if feat in num_feat_index:
                    row.append(i - 1)
                    col.append(num_feat_index[feat])
                    data.append(json_data[i][feat])
                elif feat != meta['label_name']:
                    row.append(i - 1)
                    col.append(int(float(feat)))
                    data.append(1)

        return coo_matrix(
            (np.array(data), (np.array(row), np.array(col))),
            shape=(len(json_data) - 1, feat_end_index + 1 + len(feat_name_num)))

    @classmethod
    def get_labels_from_file(cls, json_file_path):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        return json_data[0]['label_name']

    @classmethod
    def assemble_feature_to_matrix_from_file(cls, json_file_path):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)

        return Preprocessor. \
            assemble_feature_to_matrix_from_json(json_data)

    def assemble_features_to_matrix(self):
        if self.assembled_data_json is None:
            self.assemble_features_labels_to_json()

        json_data = self.assembled_data_json

        return Preprocessor. \
            assemble_feature_to_matrix_from_json(json_data)

    def assemble_features_labels_to_json(self, dst_json_file_path=None):
        """
        Assemble the resulting categorical and numerical features and their corresponding
        labels into a json data. All the categorical feature will be encoded using an
        one-hot scheme.

        Parameters
        ----------
        :param dst_json_file_path: str, optional (default=None)
            If not None, the assembled json data will be written into the file at the given location.

        :return: list
            Assembled json data.

        """
        # initialize
        self.assembled_data_json = []

        # meta_data
        self.assembled_data_json.append({
            'start_index_cat': 0,
            'end_index_cat': self._idx,
            'feat_name_num': self.numerical_feature_name_,
            'label_name': self.label_name_
        })

        for item in self._item_samples.keys():
            feat_label = {}
            self.assembled_data_json.append(feat_label)

            # label
            feat_label.update({self.label_name_: self._Y[self._item_samples[item][0]]})

            # categorical features
            for feat_index in self._item_samples.get(item):
                feat = self._X_cat[feat_index]
                for i in xrange(0, len(feat)):
                    feat_label.update({str(feat[i]): True})

            # numerical features
            feat_index = self._item_samples[item][0]
            feat = self._X_num[feat_index]
            for i in xrange(0, len(feat)):
                feat_label.update({self.numerical_feature_name_[i]: feat[i]})

        if dst_json_file_path is not None:
            with open(dst_json_file_path, 'w') as f:
                json.dump(self.assembled_data_json, f)

        return self.assembled_data_json

    def _encode_features(self, categorical_feature_name, numerical_feature_name, label_name):

        self.categorical_feature_name_ = categorical_feature_name
        self.numerical_feature_name_ = numerical_feature_name
        self.label_name_ = label_name

        self.__encode_categorical_values(self.jsondata_, categorical_feature_name)

        row_idx = 0
        for d in self.jsondata_:

            sample_cat = []
            self._X_cat.append(sample_cat)
            sample_num = []
            self._X_num.append(sample_num)

            # map sample to their row indices
            if self._item_samples.has_key(d.get('item')):
                self._item_samples.get(d.get('item')).append(row_idx)
            else:
                self._item_samples.update({d.get('item'): [row_idx]})
            row_idx += 1

            # map sample to its encoded feature values
            if self._item_values.has_key(d.get('item')):
                values = self._item_values.get(d.get('item'))
            else:
                values = {}
                self._item_values.update({d.get('item'): values})

            # append labels
            self._Y.append(d[self.label_name_])

            # append categorical features
            for key in categorical_feature_name:
                if d.has_key(key):
                    val = self.cat_value_encode_dict.get(key).get(d.get(key))
                    if not values.has_key(val):
                        sample_cat.append(val)
                        values.update({val: None})
                    else:
                        sample_cat.append(np.nan)
                else:
                    sample_cat.append(np.nan)

            # append numerical feature
            for key in numerical_feature_name:
                if d.has_key(key):
                    val = float(d[key])
                    encoded_val = key + str(val)
                    if not values.has_key(key):
                        sample_num.append(val)
                        values.update({key: encoded_val})
                    else:
                        if values.get(key) == encoded_val:
                            sample_num.append(val)
                        else:
                            raise ValueError(d['item'] + ": Detect multi-valued numerical feature: " + key +
                                             "[" + values.get(key) + " " + encoded_val + "]")
                else:
                    sample_num.append(np.nan)

    def __encode_categorical_values(self, jsondata, categorical_feature_name):
        # initialize category map
        for key in categorical_feature_name:
            self.cat_value_encode_dict.update({key: {}})

        # encode categorical values
        for key in categorical_feature_name:
            for d in jsondata:
                if d.has_key(key):
                    attr = d.get(key)
                    if not self.cat_value_encode_dict.get(key).has_key(attr):
                        self._idx += 1
                        self.cat_value_encode_dict.get(key).update({attr: self._idx})


class JsonManipulator:
    """
        A Meta Class providing an abstract method to process json data

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def manipulate_json_data(self, json_data):
        """
        The initializer of Preprocessor will call this method after load json
        data from file.

        :param json_data: list
            The target json data
        """
        raise NotImplementedError



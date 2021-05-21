from Pipeline.models import models as md

import numpy as np
import pandas as pd
import os



class AutoCorrelation(object):

    def __init__(self, train_data, validation_rule, rule, all_features, corr_factor):

        self._train_data = train_data
        self._validation_rule = validation_rule
        self._rule = rule
        self._all_features = all_features
        self._corr_factor = corr_factor
        self._high_corr_relationships = dict()
        self._features = list()
        self._abs_corr = None
        self._apply_validation_rule()
        self._apply_rule()
        self._calculate_correlations()
        self._calculate_corr_relationships()
        self._get_first_list()
        self._include_features_list()

    def _apply_rule(self):

        # Apply rule in train data and validation data
        train_with_rule = md.RuleApplier(
            self._train_data,
            rule=self._rule
        )

        self._train_data = train_with_rule.transformed_data

    def _apply_validation_rule(self):

        train_data = md.ValidationRuleApplier(
            self._train_data,
            validation_rule=self._validation_rule, 
            key='train'
        )

        self._train_data = train_data.cut_data

    def _calculate_correlations(self):

        self._abs_corr = np.abs(self._train_data[self._all_features].corr())

    def _calculate_corr_relationships(self):

        for feature in self._all_features:

            ordered_corr_series = self._abs_corr[feature].sort_values(ascending=False)
            multicorr_list = self._abs_corr[feature]\
                [(ordered_corr_series > self._corr_factor) & (ordered_corr_series.index != feature)]\
                    .index.tolist()
            self._high_corr_relationships[feature] = multicorr_list

        return None

    def _get_first_list(self):

        first_list = []
        values = []

        for key, value in self._high_corr_relationships.items():

            values += value 
            first_list = first_list + [key] if key not in values else first_list

        
        self._features.append(first_list)

    def _include_features_list(self):

        for feature in self._features[0]:

            new_list = list(self._features[0])
            index = new_list.index(feature)
            new_list.remove(feature)

            for new_feature in self._high_corr_relationships.get(feature):
                
                new_list.insert(index, new_feature)
                self._features.append(list(new_list))
                new_list.remove(new_feature)

    @property
    def features(self):

        return self._features

    @property
    def high_corr_relationships(self):

        return self._high_corr_relationships

    @property
    def train_data(self):

        return self._train_data

    @property
    def abs_corr(self):

        return self._abs_corr

class FeatureSelectionAnlysis(object):
    """
    Class responsible for run models for different input combinations, generating a complete
    dataframe with all user defined metrics.
    """
    def __init__(self, features, model_info, metrics, model_type='regressor'):
        """
        args:
        -----
        features (list) -> list of lists where each inner list represent a group of inputs
            to be used to generate a model and calculate metrics.
        model_info (dict) -> all information need to perform model calculation.
        metrics (list) -> list of dicts where each dict describe a specific metric, with 
            the name and the function used to calculate.
        model_type (str) -> define model to perform and instantiate the right classes.
        """
        self._features = features
        self._model_info = model_info
        self._metrics = metrics
        self._model_info_class_name = model_type.capitalize() + "Info"
        self._mount_model_class_name = 'Mount' + model_type.capitalize()
        self._feature_selection_info = None
        self._iter_range = iter(range(len(self._features)))
        self._predictions = pd.DataFrame()
        self._input_metrics_information()

    def _mount_base_dataframe(self):

        data = {
            'id': [np.nan],
            'features': [np.nan],
            'train points': [np.nan],
            'validation points': [np.nan],
        }

        for metric in self._metrics:

            data[metric['name']] = [np.nan]

        base_dataframe = pd.DataFrame(data=data)

        return base_dataframe

    def _mount_predictions_info(self, id_model, prediction):

        self._predictions[f'id - {id_model}'] = prediction

        return None

    def _input_metrics_information(self):

        base_dataframe = self._mount_base_dataframe()
        
        for features in self._features:

            id_model = next(self._iter_range)

            data = dict()

            self._model_info['features'] = features
            model_info = eval(f'md.{self._model_info_class_name}(**self._model_info)')
            model = eval(f'md.{self._mount_model_class_name}(model_info)')

            data['id'] = int(id_model)
            data['features'] = features 
            data['train points'] = len(model.train_data)
            data['validation points'] = len(model.validation_data)

            for metric in self._metrics:

                model.add_metric(metric)
                
                data[metric['name']] = eval(f'model.{metric["name"]}')

            base_dataframe = base_dataframe.append(data, ignore_index=True)

            self._mount_predictions_info(id_model, model.validation_data['prediction'])

        # In order to keep column id iquals column names of predictions dataset  
        # Type conversion was applied. float -> int.
        self._feature_selection_info = base_dataframe.dropna()
        self._feature_selection_info =\
            self._feature_selection_info.astype({'id': 'int64'})

    def save(self, path, filename):
        """
        Save files containing needed info to evaluate feature selection info. Two files
        are saved, one containing relationships between groups of features and metrics and
        another containing all predictions.

        args:
        -----
        path (str) -> path to save feature selection files.
        filename (str) -> name used for files.
        """
        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        feature_selection_path = ''.join([path, 'metrics_', filename])
        predictions_path = ''.join([path, 'predictions_', filename])

        print(f'saving {feature_selection_path}...')
        print(f'saving {predictions_path}...')

        self._feature_selection_info.to_csv(feature_selection_path, sep=';', index=False)
        self._predictions.to_csv(predictions_path, index=False)


    @property 
    def predictions(self):

        return self._predictions

    @property
    def feature_selection_info(self):

        return self._feature_selection_info

    @property
    def model_info(self):

        return self._model_info

    @property
    def model_info_class_name(self):

        return self._model_info_class_name
    
    @property
    def mount_model_class_name(self):

        return self._mount_model_class_name


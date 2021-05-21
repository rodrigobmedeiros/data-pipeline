from Pipeline.models import models as md

import numpy as np
import pandas as pd
from itertools import product
import os

class HyperTunning(object):

    def __init__(self, base_params, params_range, model_info, metrics, model_type='regressor'):

        self._base_params = base_params 
        self._params_range = params_range
        self._model_info = model_info
        self._metrics = metrics
        self._model_info_class_name = model_type.capitalize() + "Info"
        self._mount_model_class_name = 'Mount' + model_type.capitalize()
        self._iter_range = None
        self._all_params_combinations = None
        self._hyperparam_tunning_results = None
        self._predictions = pd.DataFrame()
        self._mount_all_combinations()
        self._mount_metrics_info()

    def _mount_all_combinations(self):

        params_list = [self._params_range[key] for key in self._params_range.keys()]
        combinations = list(product(*params_list))
        columns = self._params_range.keys()
        self._all_params_combinations = pd.DataFrame(data=combinations, columns=columns)

        self._iter_range = iter(range(len(self._all_params_combinations)))

        return None

    def _mount_base_dataframe(self, columns):

        base_dataframe = pd.DataFrame(columns=columns)

        return base_dataframe

    def _mount_predictions_info(self, id_model, prediction):

        self._predictions[f'{id_model}'] = prediction

        return None

    def _mount_metrics_info(self):

        var_params = list(self._params_range.keys())
        metrics = [metric.get('name') for metric in self._metrics]

        self._hyperparam_tunning_results = self._mount_base_dataframe(['id'] + var_params + metrics)

        for index, row in self._all_params_combinations.iterrows():
            
            variable_parameters = dict(row)
            params = dict(**variable_parameters, **self._base_params)
            params['n_estimators'] = int(params['n_estimators'])

            self._model_info['model']['params'] = params

            model_info = eval(f'md.{self._model_info_class_name}(**self._model_info)')
            model = eval(f'md.{self._mount_model_class_name}(model_info)')

            data = dict()
            data['id'] = next(self._iter_range)

            for metric in self._metrics:

                model.add_metric(metric)
                
                data[metric['name']] = eval(f'model.{metric["name"]}')

            data = dict(**data, **variable_parameters)

            self._hyperparam_tunning_results =\
                    self._hyperparam_tunning_results.append(data, ignore_index=True)

            self._mount_predictions_info(f'id - {data["id"]}', model.validation_data['prediction'])

        # Include Resultado into predictions_info 
        self._mount_predictions_info('Resultado', model.validation_data['Resultado'])

        # Convert id column type just to equal id column with each value of
        # predictions dataframe.
        self._hyperparam_tunning_results =\
            self._hyperparam_tunning_results.astype({'id': 'int64'})

        return None 

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

        hyper_tunning_results_path = ''.join([path, 'metrics_', filename])
        predictions_path = ''.join([path, 'predictions_', filename])

        print(f'saving {hyper_tunning_results_path}...')
        print(f'saving {predictions_path}...')

        self._hyperparam_tunning_results.to_csv(hyper_tunning_results_path, sep=';', index=False)
        self._predictions.to_csv(predictions_path, index=False)

    @property 
    def predictions(self):

        return self._predictions

    @property 
    def hyperparam_tunning_results(self):

        return self._hyperparam_tunning_results

    @property
    def all_params_combinations(self):

        return self._all_params_combinations


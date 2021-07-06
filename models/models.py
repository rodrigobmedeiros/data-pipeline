import pandas as pd
import numpy as np
import seaborn as sns
import pickle as pkl
import json
import dill as pkl
import itertools
import os

from functools import wraps
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt

class ClassifierInfo(object):

    def __init__(self,
                 name=None,
                 validation_rule=None,
                 classification_rule=None, 
                 non_inter_data=None,
                 inter_data=None,
                 scaler=None,
                 model=None, 
                 metrics_rule=None,
                 features=None, 
                 target=None,
                 feature_engineering=None,
                 model_import_string=None):

        self.name = name 
        self.classification_rule = classification_rule 
        self.validation_rule = validation_rule
        self.non_inter_data = non_inter_data
        self.inter_data = inter_data
        self.scaler = scaler 
        self.model = model 
        self.metrics_rule = metrics_rule
        self.features = features
        self.target = target
        self.feature_engineering = feature_engineering
        self.model_import_string = model_import_string

class ValidationRuleApplier(object):
    """
    Class responsible for apply validation rule, capable to identify if the dataset will be
    used to train ou validation.
    """
    def __init__(self, data, key, validation_rule):
        """
        To initialize instances, this class receives data information and the rules to cut
        data considering if it will be used for train or validation.

        args: 
        -----
        data (pd.DataFrame) -> original data in a pandas dataframe format. 
        key (string) -> string used to identify which part of data will be returned. 
                        train -> return training data (defined into validation rule)
                        validation -> return validation data (also defined into validation rule)
        validation_rule (function) -> user defined function that define what will be considered
                                      train or validation data.
        """
        self._original_data = data
        self._key = key 
        self._validation_rule = validation_rule  
        self._cut_data = None
        self._apply_validation_rule()

    def _apply_validation_rule(self): 
        """
        Apply validation rule to cut data considering the key that identifies what will be
        returned between train and validation part.
        """
        self._cut_data = self._validation_rule(self._original_data, self._key)

    @property
    def cut_data(self): 
        """
        Expose data to external use.
        """
        return self._cut_data

class ClassificationRuleApplier(object):
    """
    Class responsible for apply classification rule.
    """
    def __init__(self, data, classification_rule):
        """
        To initialize instances, this class receive the original data and a function that
        defines the classification rule, exposing a transformed dataset to be used by other
        features in the pipeline.

        args:
        ----- 
        data (pd.DataFrame) -> original dataset, used as reference to include the "classes" 
                               column.
        classification_rule (function) -> function used to define the classification rule, 
                                          defining all classes considered for a specific classifier.
        """
        self._original_data = data 
        self._classification_rule = classification_rule
        self._transformed_data = None
        self._apply_classification_rule()
    
    def _apply_classification_rule(self): 
        """
        Applies the function with the classification rule.
        """
        self._transformed_data = self._classification_rule(self._original_data)

        return None

    @property
    def transformed_data(self):
        """
        Expose data to external use.
        """

        return self._transformed_data

class RuleApplier(object):
    """
    Class responsible for apply classification rule.
    """
    def __init__(self, data, rule):
        """
        To initialize instances, this class receive the original data and a function that
        defines the classification rule, exposing a transformed dataset to be used by other
        features in the pipeline.

        args:
        ----- 
        data (pd.DataFrame) -> original dataset, used as reference to filter using regression rule.
        regression_rule (function) -> function used to define the regression rule, filtering just 
                                      rows included in a specific regressor.
        """
        self._original_data = data 
        self._rule = rule
        self._transformed_data = None
        self._apply_regression_rule()
    
    def _apply_regression_rule(self): 
        """
        Applies the function with the classification rule.
        """
        self._transformed_data = self._rule(self._original_data)

        return None



    @property
    def transformed_data(self):
        """
        Expose data to external use.
        """

        return self._transformed_data

class NanDiscarder(object): 
    """
    Class responsible for remove nan values considering a base dataset.
    """
    def __init__(self, data):
        """
        args:
        ----- 
        data (pd.DataFrame) -> original data containing nans to be removed.
        """
        self._data = data 
        self._data_without_nan = None
        self._discard_nan()

    def _discard_nan(self):
        """
        Remove nan from original data storing the result inside data_without_nan.
        """
        self._data_without_nan = self._data.dropna()

        return None 
    
    @property
    def data_without_nan(self):
        """
        Expose data for external use
        """
        return self._data_without_nan

class FitScaler(object):
    """
    Class used to fit a scaler object.
    """
    def __init__(self, data, scaler): 
        """
        args: 
        ----- 
        data (pd.DataFrame) -> original data filtered by the right columns 
        scaler (scaler object) ->  User define model to be fitted.
        """
        self._data = data 
        self._scaler = scaler
        self._fitted_scaler = None
        self._fit_scaler()
        
    def _fit_scaler(self):
        """
        Fit the scaler object using data and support to remove nan values.
        """
        self._fitted_scaler = self._scaler.fit(self._data)


        return None

    @property
    def fitted_scaler(self): 
        """
        Expose fitted scaler for external use.
        """
        return self._fitted_scaler

class FitModel(object):
    """
    Class used to fit a model object
    """
    def __init__(self, data, features, target, model):
        """
        args:
        ----- 
        data (pd.DataFrame) -> Data containg all needed features and target column 
        features (list) -> List of features used to fit model.
        target (string) -> Column name used as the model target  
        model (dict) -> Dictionary containing a instance and the params of the model.
                        The instance must follows sklearn pattern.
        """
        self._data = data
        self._features = features
        self._target = target
        self._model = model
        self._model_instance = self._model.get('instance')
        self._model_params = self._model.get('params')
        self._fit_model()
    
    def _fit_model(self):
        """
        Fit model using user defined params and data, including support to remove nans.
        """
        self._model_instance.set_params(**self._model_params)
        self._model_instance.fit(self._data[self._features],
                                 self._data[self._target])

        return None   

    @property 
    def fitted_model(self):
        """
        Expose fitted model for external use.
        """
        return self._model_instance

class CreatePipeline(object):
    """
    Create pipelines containing scaler and model objects with their respective names.
    """
    def __init__(self, pipeline_steps: list):
        """
        args:
        ----- 
        pipeline_steps (list) -> list of tuples containg each pipeline step with the name and
                                 the object used to transform data.
        """
        self._pipeline_steps = pipeline_steps
        self._pipeline = None
        self._mount_pipeline()

    def _mount_pipeline(self): 
        """
        Mount pipeline considering a list of tuples, containing the name and object to apply 
        data transformation.
        """
        self._pipeline = Pipeline(self._pipeline_steps)

    @property
    def pipeline(self):
        """
        Expose model pipeline to external use.
        """
        return self._pipeline

class MetadataInfo(object):
    """
    Class used to create a data structure containing all information needed to replicate 
    a model.
    """
    def __init__(self, 
                 inputs=None, 
                 train_data=None, 
                 validation_data=None, 
                 fitted_scaler=None, 
                 fitted_model=None, 
                 pipeline=None):
        """
        args:
        ----- 
        inputs (ClassificationInfo) -> instance containing all input information.
        train_data (pd.DataFrame) -> non-interpolated data used to train the model.
        validation_data (pd.dataFrame) -> interpolated data used to validate the model.
        fitted_scaler (scaler) -> instance used to normalize train data.
        fitted_model (model) -> instance used to make predictions.
        pipeline (Pipeline) -> pipeline containing all steps needed to make predictions.
        """
        self._inputs = inputs  
        self._train_data = train_data
        self._validation_data = validation_data
        self._fitted_model =fitted_model
        self._pipeline = pipeline
        self._fitted_scaler = fitted_scaler
        self._complete_information = None 

    def fit(self, inputs, train_data, validation_data, fitted_scaler, fitted_model, pipeline):
        """
        Fit all atributes before generate complete information.
        args:
        ----- 
        inputs (ClassificationInfo) -> instance containing all input information.
        train_data (pd.DataFrame) -> non-interpolated data used to train the model.
        validation_data (pd.dataFrame) -> interpolated data used to validate the model.
        fitted_scaler (scaler) -> instance used to normalize train data.
        fitted_model (model) -> instance used to make predictions.
        pipeline (Pipeline) -> pipeline containing all steps needed to make predictions.
        """
        self._inputs = inputs  
        self._train_data = train_data
        self._validation_data = validation_data
        self._fitted_model =fitted_model
        self._pipeline = pipeline
        self._fitted_scaler = fitted_scaler

        return None 

    @property
    def inputs(self):

        return self._inputs 
    
    @property
    def train_data(self):

        return self._train_data 

    @property
    def validation_data(self):

        return self._validation_data

    @property
    def fitted_scaler(self):

        return self._fitted_scaler

    @property
    def fitted_model(self):

        return self._fitted_model 

    @property
    def pipeline(self):

        return self._pipeline
        
class MountClassifier(object):

    def __init__(self, class_info: ClassifierInfo):
        
        self._classifier_info = class_info
        self._total_features = None
        self._base_features = None
        self._train_data = None 
        self._train_data_scaled = None
        self._validation_data = None 
        self._fitted_scaler = None 
        self._fitted_model = None 
        self._pipeline = None
        self._cm = None
        self._metadata = MetadataInfo()
        self._apply_validation_rule()
        self._apply_feature_engineering()
        self._apply_classification_rule()
        self._mount_features()
        self._prepare_train_data()
        self._fit_scaler()
        self._fit_model()
        self._mount_pipeline()
        self._prepare_validation_data()
        self._apply_pipeline()
        self._confusion_matrix()

    def _apply_validation_rule(self):


        train_data = ValidationRuleApplier(self._classifier_info.non_inter_data,
                     key='train',
                     validation_rule=self._classifier_info.validation_rule)
        
        validation_data = ValidationRuleApplier(self._classifier_info.inter_data,
                          key='validation',
                          validation_rule=self._classifier_info.validation_rule)
        
        self._train_data = train_data.cut_data
        self._validation_data = validation_data.cut_data

        return None
    
    def _apply_classification_rule(self): 

        # Apply classifier rule in train data
        train_with_classes = ClassificationRuleApplier(self._train_data, 
                             classification_rule=self._classifier_info.classification_rule)

        validation_with_classes = ClassificationRuleApplier(self._validation_data, 
                                  classification_rule=self._classifier_info.classification_rule)

        self._train_data = train_with_classes.transformed_data
        self._validation_data = validation_with_classes.transformed_data 

        return None

    def _apply_feature_engineering(self):

        train_with_rule = RuleApplier(self._train_data, 
                                      rule=self._classifier_info.feature_engineering.get('function'))

        validation_with_rule = RuleApplier(self._validation_data, 
                                     rule=self._classifier_info.feature_engineering.get('function'))

        self._train_data = train_with_rule.transformed_data
        self._validation_data = validation_with_rule.transformed_data 

        return None

    def _mount_features(self):

        features_to_train = self._classifier_info.features
        calculated_features_mapper = self._classifier_info.feature_engineering.get('mapper')

        mapper_keys = list(calculated_features_mapper.keys())
        mapper_values = sum(calculated_features_mapper.values(), [])

        self._total_features = list(set(features_to_train + mapper_values))
        self._base_features = list(set(self._total_features) - set(mapper_keys))

        return None

    def _prepare_train_data(self):

        self._train_data = self._train_data[self._classifier_info.features +\
                                [self._classifier_info.target]]

        self._train_data = NanDiscarder(self._train_data).data_without_nan

        return None

    def _fit_scaler(self):

        fit_scaler = FitScaler(self._train_data[self._classifier_info.features],
                               self._classifier_info.scaler)
        self._fitted_scaler = fit_scaler.fitted_scaler
        self._train_data_scaled = self._train_data.copy()

        train_data_transformed = self._fitted_scaler.\
            transform(self._train_data[self._classifier_info.features])

        self._train_data_scaled.\
            loc[:, self._classifier_info.features] = train_data_transformed

        return None

    def _fit_model(self):


        fit_model = FitModel(self._train_data_scaled, 
                    self._classifier_info.features, 
                    self._classifier_info.target,
                    self._classifier_info.model)

        self._fitted_model = fit_model.fitted_model

    def _mount_pipeline(self):

        scaler = ('scaler', self._fitted_scaler) 
        model = ('model', self._fitted_model)
        pipeline = CreatePipeline([scaler, model]) 
        self._pipeline = pipeline.pipeline   

        return None

    def _prepare_validation_data(self):

        self._validation_data = self._validation_data[self._classifier_info.features +\
                                [self._classifier_info.target]]

        self._validation_data = NanDiscarder(self._validation_data).data_without_nan

        return None

    def _apply_pipeline(self): 

        self._validation_data['prediction'] =\
        self._pipeline.predict(self._validation_data[self._classifier_info.features])

        return None

    def _confusion_matrix(self):

        self._cm = confusion_matrix(
            self._validation_data[self._classifier_info.target],
            self._validation_data['prediction'],            
        )

    def plot_confusion_matrix(self, 
                              normalize=True, 
                              title='Confusion Matrix',
                              cmap=plt.cm.Blues,
                              path='',
                              save=False):
        """
        This function prints and plots the confusion matrix. Normalization can be  
        applied by setting normalize=True.

        args:
        -----  
        normalize (bool) -> Setting normalize True present normalized values into 
            confusion matrix.
        title (str) -> User defined title inserted into plot.
        cmap (plt.cm) -> Color Map used to show the confusion matrix.
        """
        classes = sorted(self._validation_data[self._classifier_info.target].unique())
        accuracy = np.trace(self._cm) / float(np.sum(self._cm))  
        misclass = 1 - accuracy    

        cm = self._cm

        if normalize:

            cm = self._cm.astype('float') / self._cm.sum(axis=1)[:, np.newaxis]
            print('Normalized confusion matrix')

        else:

            print('Confusion matrix without normalization')  

        plt.figure(figsize=(8,8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)  
        plt.title(title)
        plt.colorbar() 
        tick_marks = np.arange(len(classes))  
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

            plt.text(
                j, i, format(cm[i, j], fmt),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

        plt.tight_layout()
        plt.ylabel('Real')
        plt.xlabel('Condição de processo predita')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy, misclass
        ))

        if save:

            if not os.path.exists(path):

                os.makedirs(path)
                print(f'directory created {path}')

            plt.savefig(''.join([path, 'confusion_matrix.png']))

        plt.show()

    def save_models(self, path):
        """
        Save models in pickle format into path passed as parameter.

        args: 
        ----- 
        path (str) -> path to save scaler and model.
        """

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        scaler_filename = f'{self._classifier_info.name}_scaler.m'
        model_filename = f'{self._classifier_info.name}_model.m'
        pipeline_filename = f'{self._classifier_info.name}_pipeline.m'

        scaler_path = ''.join([path, scaler_filename])
        model_path = ''.join([path, model_filename])
        pipeline_path = ''.join([path, pipeline_filename])

        print(f'saving scaler: {scaler_filename}')
        print(f'saving model: {model_filename}')
        print(f'saving pipeline: {pipeline_filename}')
        pkl.dump(self._fitted_scaler, open(scaler_path, 'wb'))
        pkl.dump(self._fitted_model, open(model_path, 'wb'))
        pkl.dump(self._pipeline, open(pipeline_path, 'wb'))

        return None 

    def save_metadata(self, path):

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        reduced_class_info = ClassifierInfo(name=self._classifier_info.name, 
                                                features=self._classifier_info.features, 
                                                inter_data=self._classifier_info.inter_data, 
                                                non_inter_data=self._classifier_info.non_inter_data,
                                                scaler=self._classifier_info.scaler,
                                                model=self._classifier_info.model,
                                                target=self._classifier_info.target)

        self._metadata.fit(inputs=reduced_class_info,
                           train_data=self._train_data,
                           validation_data=self._validation_data,
                           fitted_scaler=self._fitted_scaler, 
                           fitted_model=self._fitted_model,
                           pipeline=self._pipeline)

        filename = f'{self._classifier_info.name}_metadata.m'
        metadata_path = ''.join([path, filename])

        print('saving metadata...')
        pkl.dump(self._metadata, open(metadata_path, 'wb'))

        return None

    def save_32_bits_config(self, path):

        print('Saving 32 bits environment...')

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        path_32_bits = ''.join([path, "32_bits\\"])

        if not os.path.exists(path_32_bits):

            os.makedirs(path_32_bits)
            print(f'directory created {path_32_bits}')

        dataset_filename = ''.join([self._classifier_info.name, '_train_data.csv'])

        print('saving dataset to train the model...') 
        self._train_data.to_csv(''.join([path_32_bits, dataset_filename]))

        
        model_info_32_bits = {
            'name': self._classifier_info.name,
            'target_column': self._classifier_info.target,
            'features': self._classifier_info.features,
            'train_data': dataset_filename,
            'scaler': {
                'instance': str(self._classifier_info.scaler),
                'import-string': self._classifier_info.model_import_string['scaler']
            },
            'model': {
                'instance': str(self._classifier_info.model['instance']),
                'params': self._classifier_info.model['params'],
                'import-string': self._classifier_info.model_import_string['model']
            }
        }

        json_filename = ''.join([self._classifier_info.name, '_32_bits.json'])

        print('saving json config for 32 bits...')

        with open(''.join([path_32_bits, json_filename]), 'w') as f:

            json.dump(model_info_32_bits, f, indent=4)

        return None

    @property  
    def base_features(self):

        return self._base_features

    @property
    def train_data(self):

        return self._train_data

    @property
    def validation_data(self):

        return self._validation_data

    @property
    def fitted_scaler(self):

        return self._fitted_scaler

    @property
    def fitted_model(self):

        return self._fitted_model

    @property  
    def pipeline(self): 

        return self._pipeline

    @property 
    def classifier_info(self):

        return self._classifier_info  

class RegressorInfo(object):

    def __init__(self,
                 name=None,
                 validation_rule=None,
                 regression_rule=None, 
                 non_inter_data=None,
                 inter_data=None,
                 scaler=None,
                 model=None, 
                 metrics_rule=None,
                 features=None, 
                 target=None,
                 feature_engineering=None,
                 model_import_string=None):

        self.name = name 
        self.regression_rule = regression_rule 
        self.validation_rule = validation_rule
        self.non_inter_data = non_inter_data
        self.inter_data = inter_data
        self.scaler = scaler 
        self.model = model 
        self.metrics_rule = metrics_rule
        self.features = features
        self.target = target
        self.feature_engineering = feature_engineering
        self.model_import_string = model_import_string

class MountRegressor(object):

    def __init__(self, regressor_info: RegressorInfo):
        
        self._regressor_info = regressor_info
        # self._classifier_info = None
        self._total_features = None
        self._base_features = None
        self._train_data = None 
        self._train_data_scaled = None
        self._validation_data = None 
        self._fitted_scaler = None 
        self._fitted_model = None 
        self._pipeline = None
        self._metrics = set()
        self._metadata = MetadataInfo()
        self._apply_validation_rule()
        self._apply_regression_rule()
        self._apply_feature_engineering()
        self._mount_features()
        self._prepare_train_data()
        self._fit_scaler()
        self._fit_model()
        self._mount_pipeline()
        self._prepare_validation_data()
        self._apply_pipeline()

    def _apply_validation_rule(self):


        train_data = ValidationRuleApplier(self._regressor_info.non_inter_data,
                     key='train',
                     validation_rule=self._regressor_info.validation_rule)
        
        validation_data = ValidationRuleApplier(self._regressor_info.inter_data,
                          key='validation',
                          validation_rule=self._regressor_info.validation_rule)
        
        self._train_data = train_data.cut_data
        self._validation_data = validation_data.cut_data

        return None
    
    def _apply_regression_rule(self): 

        # Apply rule in train data and validation data
        train_with_rule = RuleApplier(self._train_data, 
                                      rule=self._regressor_info.regression_rule)

        validation_with_rule = RuleApplier(self._validation_data, 
                                           rule=self._regressor_info.regression_rule)

        self._train_data = train_with_rule.transformed_data
        self._validation_data = validation_with_rule.transformed_data 

        return None

    def _apply_feature_engineering(self):

        train_with_rule = RuleApplier(self._train_data, 
                                     rule=self._regressor_info.feature_engineering.get('function'))

        validation_with_rule = RuleApplier(self._validation_data, 
                                     rule=self._regressor_info.feature_engineering.get('function'))

        self._train_data = train_with_rule.transformed_data
        self._validation_data = validation_with_rule.transformed_data 

        return None

    def _mount_features(self):

        features_to_train = self._regressor_info.features
        calculated_features_mapper = self._regressor_info.feature_engineering.get('mapper')

        mapper_keys = list(calculated_features_mapper.keys())
        mapper_values = sum(calculated_features_mapper.values(), [])

        self._total_features = list(set(features_to_train + mapper_values))
        self._base_features = list(set(self._total_features) - set(mapper_keys))

        return None

    def _prepare_train_data(self):

        self._train_data = self._train_data[self._regressor_info.features +\
                                [self._regressor_info.target]]

        self._train_data = NanDiscarder(self._train_data).data_without_nan

        return None

    def _fit_scaler(self):

        fit_scaler = FitScaler(self._train_data[self._regressor_info.features],
                               self._regressor_info.scaler)
        self._fitted_scaler = fit_scaler.fitted_scaler
        self._train_data_scaled = self._train_data.copy()

        train_data_transformed = self._fitted_scaler.\
            transform(self._train_data[self._regressor_info.features])

        self._train_data_scaled.\
            loc[:, self._regressor_info.features] = train_data_transformed

        return None

    def _fit_model(self):


        fit_model = FitModel(self._train_data_scaled, 
                    self._regressor_info.features, 
                    self._regressor_info.target,
                    self._regressor_info.model)

        self._fitted_model = fit_model.fitted_model

    def _mount_pipeline(self):

        scaler = ('scaler', self._fitted_scaler) 
        model = ('model', self._fitted_model)
        pipeline = CreatePipeline([scaler, model]) 
        self._pipeline = pipeline.pipeline   

        return None

    def _prepare_validation_data(self):

        self._validation_data = self._validation_data[self._regressor_info.features +\
                                [self._regressor_info.target]]

        self._validation_data = NanDiscarder(self._validation_data).data_without_nan

        return None

    def _apply_pipeline(self): 

        self._validation_data['prediction'] =\
        self._pipeline.predict(self._validation_data[self._regressor_info.features])

        return None

    def save_models(self, path):
        """
        Save models in pickle format into path passed as parameter.

        args: 
        ----- 
        path (str) -> path to save scaler and model.
        """

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        scaler_filename = f'{self._regressor_info.name}_scaler.m'
        model_filename = f'{self._regressor_info.name}_model.m'
        pipeline_filename = f'{self._regressor_info.name}_pipeline.m'

        scaler_path = ''.join([path, scaler_filename])
        model_path = ''.join([path, model_filename])
        pipeline_path = ''.join([path, pipeline_filename])

        print(f'saving scaler: {scaler_filename}')
        print(f'saving model: {model_filename}')
        print(f'saving pipeline: {pipeline_filename}')
        pkl.dump(self._fitted_scaler, open(scaler_path, 'wb'))
        pkl.dump(self._fitted_model, open(model_path, 'wb'))
        pkl.dump(self._pipeline, open(pipeline_path, 'wb'))

        return None 

    def save_metadata(self, path):

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        reduced_class_info = ClassifierInfo(name=self._regressor_info.name, 
                                                features=self._regressor_info.features, 
                                                inter_data=self._regressor_info.inter_data, 
                                                non_inter_data=self._regressor_info.non_inter_data,
                                                scaler=self._regressor_info.scaler,
                                                model=self._regressor_info.model,
                                                target=self._regressor_info.target)

        self._metadata.fit(inputs=reduced_class_info,
                           train_data=self._train_data,
                           validation_data=self._validation_data,
                           fitted_scaler=self._fitted_scaler, 
                           fitted_model=self._fitted_model,
                           pipeline=self._pipeline)

        filename = f'{self._regressor_info.name}_metadata.m'
        metadata_path = ''.join([path, filename])

        print(f'saving metadata: {filename}')
        pkl.dump(self._metadata, open(metadata_path, 'wb'))

        return None

    def plot_validation(self, 
                        path, 
                        save=False, 
                        start_num=0,
                        end_num=-1,
                        prop=10,
                        title = '',
                        analysis_class='IF'): 

        y = self._validation_data.copy().reset_index()
        title = self._regressor_info.name
        factor = prop/100

        eixo = np.array(range(len(y[self._regressor_info.target][start_num:end_num])))

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        plt.autoscale(enable=True)
        plt.grid(True)


        # Plot real lab data
        plt.plot(y[self._regressor_info.target][start_num:end_num].values  ,
                label='Real',
                linewidth=0.5,
                linestyle='--',
                marker='o',
                markersize=2,
                color = 'Navy')
        # Plot predicted IF
        plt.plot(y['prediction'][start_num:end_num],
                label='Predito',
                linewidth=1.0,
                color = 'orangered')

        # Plot up lab error (10%)
        plt.fill_between(eixo,
                         y[self._regressor_info.target][start_num:end_num].values.ravel(),
                         (1 + factor)*y[self._regressor_info.target][start_num:end_num].values.ravel(),
                         color='dodgerblue',
                         label='Incerteza',
                         alpha = 0.2)

        # Plot down lab error (10%)
        plt.fill_between(eixo,
                         y[self._regressor_info.target][start_num:end_num].values.ravel(),
                         (1 - factor)*y[self._regressor_info.target][start_num:end_num].values.ravel(),
                         color='dodgerblue',
                         alpha = 0.2)

        plt.title(title)
        plt.xlabel('Índice do Ponto')
        plt.ylabel(analysis_class)
        plt.legend()

        if save:

            if not os.path.exists(path):

                os.makedirs(path)
                print(f'directory created {path}')

            plt.savefig(''.join([path, self._regressor_info.name, '_validation.png',]),
                        format='png')

        plt.show()

        return None

    def plot_feature_importance(self, path, save=False, title=''):

        if not isinstance(self._fitted_model, GradientBoostingRegressor):

            raise TypeError('Feature Importance only supports Gradient Boosting Regressor')

        df_fi = pd.DataFrame(
            list(
                zip(
                    self._regressor_info.features,
                    self._fitted_model.feature_importances_
                )
            ),
            columns=["Feature", "Value"]
        )
        fi_mean = df_fi.groupby(["Feature"])['Value'].aggregate(np.mean)\
            .reset_index().sort_values('Value', ascending=False)
        
        plt.figure(figsize=(20,8))
        sns.barplot(
            y='Feature',
            x="Value", 
            data=df_fi, 
            palette='viridis', 
            order=fi_mean['Feature']
        )
        plt.title("Feature Importance" + title)
        
        if save:

            if not os.path.exists(path):

                os.makedirs(path)
                print(f'directory created {path}')

            plt.savefig(
                ''.join(
                    [
                        path,
                        self._regressor_info.name,
                        '_feature_importance.png'
                    ]
                ),
                format='png'
            )

        plt.show()

        return None

    def add_metric(self, metrics_info):
        
        metric_name = metrics_info['name']
        metrics_function = metrics_info['function']

        y_real = self._validation_data[self._regressor_info.target] 
        y_pred = self._validation_data['prediction'] 

        self.__dict__[metric_name] = metrics_function(y_real, y_pred)
        self._metrics.add(metric_name)

        return None

    def save_32_bits_config(self, path):

        print('Saving 32 bits environment...')

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        path_32_bits = ''.join([path, "32_bits\\"])

        if not os.path.exists(path_32_bits):

            os.makedirs(path_32_bits)
            print(f'directory created {path_32_bits}')

        dataset_filename = ''.join([self._regressor_info.name, '_train_data.csv'])

        print('saving dataset to train the model...') 
        self._train_data.to_csv(''.join([path_32_bits, dataset_filename]))

        
        model_info_32_bits = {
            'name': self._regressor_info.name,
            'target_column': self._regressor_info.target,
            'features': self._regressor_info.features,
            'train_data': dataset_filename,
            'scaler': {
                'instance': str(self._regressor_info.scaler),
                'import-string': self._regressor_info.model_import_string['scaler']
            },
            'model': {
                'instance': str(self._regressor_info.model['instance']),
                'params': self._regressor_info.model['params'],
                'import-string': self._regressor_info.model_import_string['model']
            }
        }

        json_filename = ''.join([self._regressor_info.name, '_32_bits.json'])

        print('saving json config for 32 bits...')

        with open(''.join([path_32_bits, json_filename]), 'w') as f:

            json.dump(model_info_32_bits, f, indent=4)
            
        return None

    @property
    def metrics(self):

        return self._metrics

    @property  
    def base_features(self):

        return self._base_features

    @property
    def train_data(self):

        return self._train_data

    @property
    def validation_data(self):

        return self._validation_data

    @property
    def fitted_scaler(self):

        return self._fitted_scaler

    @property
    def fitted_model(self):

        return self._fitted_model

    @property  
    def pipeline(self): 

        return self._pipeline

    @property
    def total_features(self):
        return self._total_features

    @property 
    def regressor_info(self):

        return self._regressor_info  

    @property
    def base_features(self):
        return self._base_features

class GainClassifier(object):

    def __init__(self):

        pass

class MountClassifierUS(MountClassifier):

    def __init__(self, class_info: ClassifierInfo):
        
        super().__init__(class_info)

    def _fit_scaler(self):

        pass 

    def _fit_model(self):

        pass

    def _mount_pipeline(self):

        self._pipeline = GainClassifier()

        classifier_rule = self._classifier_info.classification_rule
        target = self._classifier_info.target
        
        def predict(df, rule=classifier_rule, target=target):

            df = classifier_rule(df)

            predictions = df[target]

            return np.array(predictions)

        self._pipeline.predict = predict

    def _apply_pipeline(self): 

        self._validation_data['prediction'] = self._validation_data[self._classifier_info.target]























































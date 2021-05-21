import numpy as np
import matplotlib.pyplot as plt

class ModelInfo(object):

    def __init__(self, model_type, pipeline, filter_function, features):

        self._model_type = model_type  
        self._pipeline = pipeline
        self._filter_function = filter_function
        self._features = features

    @property
    def model_type(self):

        return self._model_type 

    @property
    def pipeline(self):

        return self._pipeline 
    
    @property
    def filter_function(self):

        return self._filter_function  

    @property
    def features(self):

        return self._features

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

class GeneralValidation(object):

    def __init__(self, validation_data, models: list, feature_engineering, validation_rule):

        self._validation_data = validation_data
        self._models = models 
        self._feature_engineering = feature_engineering
        self._validation_rule = validation_rule
        self._class_column_count = 0
        self._validation_data[f'class_prediction_{self._class_column_count}'] = np.nan
        self._classifiers = None   
        self._regressors = None
        self._apply_validation_rule()
        self._apply_feature_engineering()
        self._get_classifiers()
        self._get_regressors()
        self._apply_classifiers()
        self._apply_regressors()

    def _apply_validation_rule(self):

        validation_data = ValidationRuleApplier(self._validation_data,
            key='validation',
                validation_rule=self._validation_rule)
        
        self._validation_data = validation_data.cut_data

        return None

    def _apply_feature_engineering(self):

        validation_with_rule = RuleApplier(self._validation_data, 
                                     rule=self._feature_engineering)

        self._validation_data = validation_with_rule.transformed_data 

        return None

    def _get_classifiers(self):

        self._classifiers = list()

        for model in self._models:

            if model.model_type == 'classifier': 

                self._classifiers.append(model)


        return None   

    def _get_regressors(self):

        self._regressors = list()

        for model in self._models:

            if model.model_type == 'regressor':

                self._regressors.append(model)

        return None

    def _get_condition(self, data, features): 

        condition = ~data[features].isna().any(axis=1)

        return condition

    def _apply_classifiers(self):

        for classifier in self._classifiers:

            condition_1 = classifier.filter_function(self._validation_data)
            condition_2 = self._get_condition(self._validation_data, classifier.features)

            self._class_column_count += 1 
            class_column = f'class_prediction_{self._class_column_count}'

            self._validation_data[class_column] = \
                self._validation_data[f'class_prediction_{self._class_column_count - 1}']

            self._validation_data.loc[condition_1 & condition_2, class_column] =\
                classifier.pipeline.predict(self._validation_data.loc[condition_1 & condition_2, \
                    classifier.features])

        return None

    def _apply_regressors(self):

        last_class_column = f'class_prediction_{self._class_column_count}'
        
        self._validation_data['prediction'] = np.nan

        for regressor in self._regressors:

            condition_1 = regressor.filter_function(self._validation_data, last_class_column)
            condition_2 = self._get_condition(self._validation_data, regressor.features)

            self._validation_data.loc[condition_1 & condition_2, 'prediction'] =\
                regressor.pipeline.predict(self._validation_data.loc[condition_1 & condition_2, \
                    regressor.features])

    def plot_validation(self, 
                        path, 
                        save=False, 
                        start_num=0,
                        end_num=-1,
                        prop=10,
                        title = '',
                        analysis_class='IF'): 

        y = self._validation_data.copy().reset_index()
        title = 'General Validation'
        factor = prop/100

        eixo = np.array(range(len(y['Resultado'][start_num:end_num])))

        fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
        fig.patch.set_facecolor('white')
        plt.autoscale(enable=True)
        plt.grid(True)


        # Plot real lab data
        plt.plot(y['Resultado'][start_num:end_num].values  ,
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
                         y['Resultado'][start_num:end_num].values.ravel(),
                         (1 + factor)*y['Resultado'][start_num:end_num].values.ravel(),
                         color='dodgerblue',
                         label='Incerteza',
                         alpha = 0.2)

        # Plot down lab error (10%)
        plt.fill_between(eixo,
                         y['Resultado'][start_num:end_num].values.ravel(),
                         (1 - factor)*y['Resultado'][start_num:end_num].values.ravel(),
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

            plt.savefig(''.join([path, 'general_validation.png',]),
                        format='png')

        plt.show()

        return None

    @property 
    def classifiers(self):

        return self._classifiers 

    @property
    def regressors(self):

        return self._regressors

    @property
    def validation_data(self):

        return self._validation_data
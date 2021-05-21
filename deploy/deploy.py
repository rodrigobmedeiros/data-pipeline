import pandas as pd
import numpy as np
import json
import os

class JoinFeatures(object):

    def __init__(self, features: list):
        """
        args:
        ----- 
        features (list) -> list of lists containing features from all models.
        """
        self._features_list = features
        self._unique_features = None
        self._join_features()
    
    def _join_features(self):
        """
        join features from all models into a unique list.
        """
        total_features = sum(self._features_list, [])
        unique_features = set(total_features)

        self._unique_features = list(unique_features)

        return None   

    @property
    def unique_features(self):
        """
        Expose unique list of features to external use.
        """
        return self._unique_features

class JsonFeature(object):

    def __init__(self, feature_name, min_value, max_value, host, 
        server, feature_type, tag, decay, delay, window):
        """
        args:
        ----- 
        feature_name (str) -> Engineering name  
        min (float) -> minimum value considering non interpolated data.   
        max (float) -> maximum value considering non interpolated data.
        host (str) -> communication pattern
        server (str) -> server used to get information. 
        type (str) -> definition  if the feature is measured or calculated. 
        tag (str) ->  tag that identifies instruments in the process plant.
        decay (float) -> decay value  
        delay (float) -> delay value  
        window (float) -> window value
        """
        self._features_name = feature_name  
        self._min = min_value 
        self._max = max_value  
        self._host = host 
        self._server = server  
        self._type = feature_type 
        self._tag = tag 
        self._decay = decay 
        self._delay = delay
        self._window = window

    def format(self):

        tag_info_format = dict()
        tag_info_format['Feature Name'] = self._features_name
        tag_info_format["Host"] = self._host
        tag_info_format["Server"] = self._server
        tag_info_format["Type"] = self._type
        tag_info_format["Tag"] = self._tag.upper()
        tag_info_format["Min"] = self._min
        tag_info_format["Max"] = self._max
        tag_info_format["Delay"] = self._delay
        tag_info_format["Window"] = self._window
        tag_info_format["Decay"] = self._decay

        return tag_info_format

class CalculatedTagProcessor(object):

    

    def __init__(self, tag_info):

        self._tag_info = tag_info

    def format(self):

        info = dict()

        global_info = self._tag_info.groupby('Feature Name').agg(pd.Series.mode)
        global_info = global_info.reset_index()[['Feature Name', 'Type']]
        info = dict(**info, **global_info.to_dict(orient='records')[0])

        # Add info used to calculate  
        # Numerator

        columns_used_to_calculate = [
            'Tag', 
            'Host', 
            'Server', 
            'Min', 
            'Max', 
            'Delay', 
            'Window'
        ]


        info['Numerator'] = self._tag_info[columns_used_to_calculate].to_dict(
            orient='records'
        )

        operation = self._tag_info.groupby('Feature Name').agg(pd.Series.mode)
        operation = operation.reset_index()[['Operation']].to_dict(orient='records')[0]
        operation = operation['Operation']

        info['Denominator'] = [
            1 if operation == 'Sum' else len(info['Numerator'])
        ]

        return info

class MeasuredTagProcessor(object):

    def __init__(self, tag_info):

        self._tag_info = tag_info

    def format(self):

        info = self._tag_info.to_dict(orient='records')[0]

        return info

class ProcessTagInfo(object):

    def __init__(self, tag_info):

        self._tag_info = tag_info
        self._tag_processor = None
        self._get_tag_processor()

    def _get_tag_processor(self):

        if len(self._tag_info) > 1:

            self._tag_processor = CalculatedTagProcessor(self._tag_info)
            return None  

        self._tag_processor = MeasuredTagProcessor(self._tag_info)

    @property
    def tag_processor(self):

        return self._tag_processor

class MountJson(object):  

    def __init__(self, 
                 tag_info: pd.DataFrame, 
                 data: pd.DataFrame, 
                 sync_groups: list,
                 features: list,
                 sync_type = 'fixo'):
        """
        args:
        -----   
        tag_info (pd.DataFrame) -> dataframe containing tag information, including tag, host,
            server, typw and name.
        data (pd.DataFrame) -> dataframe containing non-interpolated data used to get extreme
            ranges.
        sync_groups (list) -> list of syncronization groups.
        features (list) -> list containing features from all models without repetition.
        """
        self._tag_info = tag_info
        self._data = data 
        self._sync_groups = sync_groups
        self._features = features
        self._sync_type = sync_type
        self._json_features = None
        self._json_info = None
        self._filter_by_features()
        self._calculate_min_max()
        self._add_sync_parameters()
        self._mount_data()
        self._generate_json_tags_info()
        #self.#self._create_format_tags()
        self._mount_json()
        
    def _filter_by_features(self):

        self._data = self._data[self._features]

        return None

    def _calculate_min_max(self):

        min_max = self._data.describe().T[['min', 'max']]
        min_max = min_max.apply({'min': np.floor, 'max': np.ceil})
        min_max.index.name = 'Feature Name'
        min_max.columns = ['Min', 'Max']
        self._data = min_max.reset_index()

        return None

    def _mount_data(self):

        self._data = self._tag_info.merge(self._data, how='left', on='Feature Name')
        self._data['Type'] = np.where(
            (self._data['Operation'] == 'Original'),
            'Measured',
            'Calculated'
        )
        return None

    def _add_sync_parameters(self):

        if self._sync_type != 'fixo': 

            self._data['Delay'] = ''
            self._data['Window'] = '' 
            self._data['Decay'] = '' 

            return None 
        
        else: 

            features_groups = pd.DataFrame()
            features_groups['Feature Name'] = self._data['Feature Name']

            delay = []
            decay = []
            window = []

            for ind, group in enumerate(self._sync_groups):

                features_groups[group.group_name] = self._data['Feature Name'].isin(group.features)
                decay.append(group.decay)
                delay.append(group.delay.transformer)
                window.append(group.window.transformer)

            
            # finishing df with features correlating with sync groups.
            features_groups.set_index('Feature Name', inplace=True)

            # defining df correlating sync groups and their decay, delay and window.
            groups_parameters = pd.DataFrame()
            groups_parameters['Groups'] = features_groups.columns
            groups_parameters['Decay'] = decay
            groups_parameters['Delay'] = delay
            groups_parameters['Window'] = window
            groups_parameters.set_index('Groups', inplace=True)

            # Defining df correlating features and sync parameters
            # Strategy using inner product to obtain feature x right parameters.
            features_parameters = np.dot(features_groups.values, groups_parameters.values)
            features_parameters = pd.DataFrame(data=features_parameters, 
                index=features_groups.index, columns=groups_parameters.columns)


            self._data = self._data.merge(features_parameters, how='left', on='Feature Name')

            return None
    
    def _generate_json_tags_info(self):
        
        json_features = list()

        for feature in self._features:

            feature_info = self._data.loc[self._data['Feature Name'] == feature]
            process_feature = ProcessTagInfo(feature_info).tag_processor

            json_features.append(process_feature.format())

        self._json_features = json_features

        return None
    """
    def _create_format_tags(self):

        json_features = list()

        for ind, item in self._data.iterrows():

            feature = JsonFeature(item['Feature Name'],
                item['min'],
                item['max'],
                item['Host'],
                item['Server'],
                item['Type'],
                item['Tag'],
                item['decay'],
                item['delay'],
                item['window'])

            json_features.append(feature.format())

        self._json_features = json_features

        return None
    """

    def _mount_json(self):

        json_info = dict()

        json_info['TAGS'] = self._json_features

        self._json_info = json_info

        return None
    
    def save(self, path):

        json_filename = 'tags_deploy.json'

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        print(f'saving json: {json_filename}')

        with open(''.join([path, json_filename]), 'w') as file:

            json.dump(self._json_info, file, indent=4)

        return None

    @property
    def json_features(self):
        return self._json_features

    @property 
    def tag_info(self):

        return self._tag_info

    @property 
    def data(self):

        return self._data

    @property 
    def sync_groups(self):

        return self._sync_groups

    @property 
    def features(self):

        return self._features

class MountModelInfo(object):

    def __init__(self, name, features, model_type, class_name=None, deadband=1):
        """
        args:
        ----- 
        name (str) -> name of the model, usually the same of the name used in the model building.
        features (list) -> List of features used to train the model.
        model_type (str) -> define the model type between classifier and regressor.
        class_name (str) -> Name of the classifier response that calls this model.
        deadband (integer) -> deadband used to calculate the real value of a specific point.
        """
        self._name = name 
        self._pipeline_name = ''.join([name, '_pipeline.m'])
        self._class_name = class_name
        self._features = features  
        self._model_type = model_type
        self._deadband = deadband

    @property 
    def name(self):

        return self._name 

    @property 
    def pipeline_name(self):

        return self._pipeline_name 

    @property
    def features(self):

        return self._features 

    @property
    def model_type(self):

        return self._model_type

    @property 
    def deadband(self):

        return self._deadband

    @property 
    def class_name(self):

        return self._class_name

class MountScriptInput(object):

    def __init__(self, all_features, root_model, other_models):

        self._all_features = all_features
        self._all_features_names = None
        self._root_model = root_model 
        self._other_models = other_models
        self._model_structure = dict() 
        self._create_features_tags()
        self._include_features()
        self._include_root_model()
        self._include_other_models()

    def _create_features_tags(self):

        self._all_features_names = [feature.get('Feature Name') for feature in self._all_features]

        return None 

    def _include_features(self):

        self._model_structure['all_features'] = self._all_features_names

        return None  

    def _include_root_model(self):

        root_model_info = dict()
        root_model_info['name'] = self._root_model.name 
        root_model_info['pipeline'] = self._root_model.pipeline_name
        root_model_info['features'] = self._root_model.features 
        root_model_info['type'] = self._root_model.model_type 
        root_model_info['deadband'] = self._root_model.deadband 

        self._model_structure['root_model'] = root_model_info

        return None

    def _include_other_models(self):
        
        self._model_structure['models'] = dict()

        for model in self._other_models:

            model_info = dict()
            model_info['type'] = model.model_type
            model_info['pipeline'] = model.pipeline_name 
            model_info['features'] = model.features 
            model_info['deadband'] = model.deadband
            

            self._model_structure['models'][model.class_name] = model_info

        return None

    def save(self, path, json_filename='config.script.info'):

        if not os.path.exists(path):

            os.makedirs(path)
            print(f'directory created {path}')

        print(f'saving json: {json_filename}')

        with open(''.join([path, json_filename]), 'w') as file:

            json.dump(self._model_structure, file, indent=4)

        return None

    @property 
    def all_features_names(self):

        return self._all_features_names
        
    @property
    def model_structure(self):

        return self._model_structure
import pandas as pd
import numpy as np
from astropy.convolution import convolve
import re

class LabDataTreater(object):
    """
    Class used to treat lab data before interpolation process
    """
    def __init__(self, lab_data, transition_window):
        """
        Method responsible for initiate objects for this class

        args
        ---- 
        lab_data (pd.DataFrame) -> Dataframe containing lab data without outlier
        transition_window (int) -> Number of points considered transition at every start and
                                   end of the campaign.
        """
        self._lab_data = lab_data.copy()
        self._transition_window = transition_window
        self._initial_length = self._lab_data.shape[0]
        self._drop_duplicates()
        self._final_length = self._lab_data.shape[0]
        self._number_of_dropped_rows = self._initial_length - self._final_length
        self._mark_campaigns()
        self._mark_transitions()

    def _drop_duplicates(self):
        """
        Drop duplicates from lab data keeping the first occurrance.
        """

        self._lab_data = self._lab_data.loc[~self._lab_data.index.duplicated()]
        self._lab_data.sort_index()

        return None

    def _mark_campaigns(self):
        """
        Mark campaigns for each material.
        """
        grades = self._lab_data.reset_index()['Material'].copy()

        buffer = dict()
        count = dict()
        for i, g in enumerate(self._lab_data.reset_index()['Material'].unique()):
            grades.loc[grades == g] = i
            buffer[i] = list()
            count[i] = list()

        counter = 0
        last = 0
        offset = 0
        for i in grades:
            if i != last:
                buffer[last].append(list(range(offset, offset + counter)))
                count[last].append([counter])
                offset += counter
                counter = 0
            counter += 1
            last = i
        buffer[last].append(list(range(offset, offset + counter)))
        count[last].append([counter])

        self._lab_data = self._lab_data.reset_index()
        for grade in buffer.keys():
            for i, indices in enumerate(buffer[grade]):
                self._lab_data.loc[indices, 'Campanha'] = i

        
        
        return None

    def _mark_transitions(self):
        """
        Mark transitions for considering points at the start and the beginning of campaigns.
        """
        self._lab_data['Transição'] = True

        for grade in self._lab_data['Material'].unique():
            for i in self._lab_data[(self._lab_data['Material']==grade)]['Campanha'].unique():
                if self._lab_data[(self._lab_data['Material']==grade) & (self._lab_data['Campanha']==i)]['data'].count() > (self._transition_window*2):
                    start_date = self._lab_data[(self._lab_data['Material']==grade) & (self._lab_data['Campanha']==i)]['data'].nsmallest(self._transition_window).iloc[-1]
                    end_date = self._lab_data[(self._lab_data['Material']==grade) & (self._lab_data['Campanha']==i)]['data'].nlargest(self._transition_window).iloc[-1]
                    self._lab_data.loc[(self._lab_data['Material']==grade) & (self._lab_data['Campanha']==i) &
                    (self._lab_data['data']>start_date) & (self._lab_data['data'] < end_date) ,'Transição'] = False  
        
        self._lab_data.set_index('data', inplace=True)

        return None

    @property
    def lab_data(self):
        """
        Return treated lab data.
        """
        return self._lab_data

    def __repr__(self):
        """
        Print usefull informarion.
        """
        repr_text = f'Initial number of rows: {self._initial_length}\n'
        repr_text += f'Number of dropped duplicated rows: {self._number_of_dropped_rows}\n'
        repr_text += f'LabDataOrganizer(lab_data=csv file, transition_window={self._transition_window})'
        
        return repr_text

class LabDataInterpolator(object):

    def __init__(self, treated_lab_data,
                       columns_to_interpolate: list,
                       limit_time=360,
                       inter_frequency='1min'):

        self._treated_lab_data = treated_lab_data
        self._limit_time = limit_time
        self._inter_frequency = inter_frequency
        self._lab_data = self._treated_lab_data.lab_data
        self._columns_to_interpolate = columns_to_interpolate
        self._samples_before_interpolation = self._lab_data.shape[0]
        self._interpolated_lab_data = pd.DataFrame()
        self._include_datetime_as_index()
        self._alocate_info_into_interpolated_dataset()
        self._interpolate_columns()


    def _include_datetime_as_index(self):

        start_date = self._lab_data.index.min()
        end_date = self._lab_data.index.max()
        date = pd.date_range(start=start_date, end=end_date, freq=self._inter_frequency)
        self._interpolated_lab_data['data'] = date 
        self._interpolated_lab_data.set_index('data', inplace=True)

        return None

    def _alocate_info_into_interpolated_dataset(self):

        for column in self._columns_to_interpolate:

            self._interpolated_lab_data[column] = self._lab_data[column] 

        return None

    def _interpolate_columns(self):

        for column in self._columns_to_interpolate:

            dtype = re.match('[A-Za-z]*', str(self._interpolated_lab_data[column].dtype).capitalize())
            class_name = ''.join([dtype.group(),
                         'Interpolator(self._interpolated_lab_data[column], self._limit_time)'])
            interpolator = eval(class_name)
            self._interpolated_lab_data[column] = interpolator.interpolated_column

        return None

    @property
    def interpolated_lab_data(self):

        return self._interpolated_lab_data

class TimeTransformer(object): 

    def __init__(self, transformer, parameter_type='fixo'):

        self._type = parameter_type
        self._transformer = transformer

    @property
    def type(self):

        return self._type  
    
    @property
    def transformer(self):

        return self._transformer

    def __repr__(self):

        return f'TimeTransformer(type={self._type})'

class SummableDataFrame(object):

    def __init__(self, dataframe=None):

        self.dataframe = dataframe 
        self.dataframe.index.name = 'data'
    
    def __add__(self, other):

        sum_dataframe = self.dataframe.merge(other.dataframe, on='data', how='outer')

        return SummableDataFrame(sum_dataframe)

class GroupSyncInfo(object):

    def __init__(self, group_name, 
                       features, 
                       decay, 
                       delay, 
                       window):

        self.group_name = group_name 
        self.features = features 
        self.decay = decay
        self.delay = delay 
        self.window = window

class SyncAllGroups(object):

    def __init__(self, groups, sensor_data):

        self._groups = groups 
        self._group_names = [group.group_name for group in self._groups]
        self._features = [group.features for group in self._groups]
        self._decays = [group.decay for group in self._groups]
        self._delays = [group.delay for group in self._groups]
        self._windows = [group.window for group in self._groups]
        self._sensor_data = sensor_data
        self._syncronized_groups = list()
        self._sync_groups()
        self._sync_dataframe = None
        self._sync_dataframes()

    def _sync_groups(self):

        for group in self._groups:

            self._syncronized_groups.append(SyncGroup(group.group_name,
                                                      group.features,
                                                      self._sensor_data,
                                                      group.decay,
                                                      group.delay,
                                                      group.window))

        return None 

    def _sync_dataframes(self):

        self._sync_dataframe = [group.sync_dataframe for group in self._syncronized_groups]
        self._sync_dataframe = sum(self._sync_dataframe, start=SummableDataFrame(pd.DataFrame()))

    @property
    def sync_dataframe(self):

        return self._sync_dataframe.dataframe

class SyncGroup(object):

    def __init__(self, group_name: str,
                       features: list,
                       sensor_data: pd.DataFrame,
                       decay: float,
                       delay: TimeTransformer,
                       window: TimeTransformer):   

        self._group_name = group_name
        self._features = features
        self._sensor_data = sensor_data[self._features]
        self._decay = decay
        self._delay = delay
        self._window = window
        self._sync_dataframe = None 
        self._sync_data()

    def _sync_data(self):

        delay_transformer = self._delay.transformer
        window_transformer = self._window.transformer

        self._sensor_data = self._sensor_data.resample('1min').asfreq()

        w = list(np.zeros(window_transformer - 1)) + [self._decay**i for i in range(0, window_transformer)]
        kernel = (np.array(w)/sum(w))
        weighted_mean_df = pd.DataFrame()
        
        for column in self._features:
            # interpolate will result in renormalization of the kernel at each position ignoring
            # (pixels that are NaN in the image) in both the image and the kernel. ‘fill’ will replace
            # the NaN pixels with a fixed numerical value (default zero, see fill_value) prior to convolution
            # Note that if the kernel has a sum equal to zero, NaN interpolation is not possible and will raise an exception.
            weighted_mean_df[column] = convolve(self._sensor_data[column].values, kernel,
                                                nan_treatment='interpolate')

        weighted_mean_df.index = self._sensor_data.index
        weighted_mean_df = weighted_mean_df.iloc[window_transformer:-window_transformer]
        self._sync_dataframe = weighted_mean_df.shift(delay_transformer)
        self._sync_dataframe = SummableDataFrame(self._sync_dataframe)
        
        return None
    
    @property 
    def sync_dataframe(self):

        return self._sync_dataframe

class ColumnInterpolator(object):

    def __init__(self, column, limit_time):

        self._column = column
        self._limit_time = limit_time
        self._interpolated_column = None
        self._interpolate_column()
    
    def _interpolate_column(self):

        return None

    @property
    def interpolated_column(self):

        return self._interpolated_column

    def __repr__(self):

        return f'{self.__class__.__name__}({self._column.dtype}, limit_time={self._limit_time})'

class ObjectInterpolator(ColumnInterpolator):


    def _interpolate_column(self):

        ffill = self._column.fillna(method='ffill')
        bfill = self._column.fillna(method='bfill')
        self._interpolated_column = (ffill == bfill) * ffill
        self._interpolated_column.loc[self._interpolated_column == ''] = 'transição - ' +\
                                                             ffill.loc[ffill != bfill] + \
                                                             ' - ' + \
                                                             bfill.loc[ffill != bfill]
        
        return None

class FloatInterpolator(ColumnInterpolator):

    def _interpolate_column(self):

        self._interpolated_column = self._column.interpolate(limit_direction='both',
                                                             limit=self._limit_time)

class MountSyncDataset(object):
    """
    class used to combine two datasets considering one as a base and another as a complement. 
    They are combined considering a left join of the base with the complementary.
    """

    def __init__(self, base_dataset: pd.DataFrame, complementar_dataset: pd.DataFrame):

        self._base_dataset = base_dataset
        self._completar_dataset = complementar_dataset
        self._combined_dataset = None
        self._combine_datasets()

    def _combine_datasets(self):
        """
        Combine base and complementar data into an unique dataset.

        return:
        pd.DataFrame containing merged data. 
        """
        self._combined_dataset = self._base_dataset.merge(self._completar_dataset,
                                                          on='data',
                                                          how='left')

        return None

    @property 
    def combined_dataset(self):

        return self._combined_dataset

    def save(self, path_to_save, filename):
        """
        Save synchronized dataset with passed filename into passed path to save.

        args: 
        -----
        path_to_save (str) -> Directory used to sabe the synchronized dataset  
        filename (str) -> name that will be used to save the data, including the extension (.csv). 
        """
        self._combined_dataset.to_csv(''.join([path_to_save, filename]))

        print(f'file {filename} saved successfully')
        print(f'file saved into {path_to_save}')
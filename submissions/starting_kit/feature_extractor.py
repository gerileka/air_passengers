import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import datetime
import warnings
warnings.filterwarnings("ignore")

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        df = pd.read_csv(os.path.join(path, 'external_data.csv'))
        distanceDF = pd.DataFrame()
        distanceDF = df[['Departure','Arrival','Distance']] 
        distanceDF.dropna(inplace = True)
        distanceDF.reset_index(inplace=True,drop= True)
        dataEXT = df.drop(['Departure','Arrival','Distance'], axis=1)
        dataEXT.dropna(how='all',inplace=True)
        X_encoded = X_encoded.merge(distanceDF,how="left", on=['Departure', 'Arrival'])
        X_encoded= X_encoded.merge(dataEXT,how="left", left_on=['DateOfDeparture','Departure'], right_on=['Date','AirPort'],
                                   suffixes=('','_d'))
        X_encoded= X_encoded.merge(dataEXT,how="left", left_on=['DateOfDeparture','Arrival'], right_on=['Date','AirPort'],
                                   suffixes=('','_a'))
        X_encoded= X_encoded.drop(['Date','AirPort', 'City'],axis=1)
        X_encoded['Mean Humidity']=X_encoded['Mean Humidity']-X_encoded['Mean Humidity_a']
        X_encoded['Population']=X_encoded['Population']-X_encoded['Population_a']
        X_encoded['Mean TemperatureC']=X_encoded['Mean TemperatureC']-X_encoded['Mean TemperatureC_a']
        X_encoded['Mean Sea Level PressurehPa']=X_encoded['Mean Sea Level PressurehPa']-X_encoded['Mean Sea Level PressurehPa_a']
        X_encoded['Dew PointC_a']=X_encoded['Dew PointC']-X_encoded['Dew PointC_a']
        X_encoded['Mean VisibilityKm']=X_encoded['Mean VisibilityKm']-X_encoded['Mean VisibilityKm_a']
        X_encoded['Mean Wind SpeedKm/h']=X_encoded['Mean Wind SpeedKm/h']-X_encoded['Mean Wind SpeedKm/h_a']
        X_encoded['MeanDew PointC']=X_encoded['MeanDew PointC']-X_encoded['MeanDew PointC_a']
        X_encoded['WindDirDegrees']=X_encoded['WindDirDegrees']-X_encoded['WindDirDegrees_a']
        X_encoded['CloudCover']=X_encoded['CloudCover']-X_encoded['CloudCover_a']
        X_encoded = X_encoded.drop('Mean Humidity_a', axis=1)
        X_encoded = X_encoded.drop('Population_a', axis=1)
        X_encoded = X_encoded.drop('Mean TemperatureC_a', axis=1)
        X_encoded = X_encoded.drop('Mean Sea Level PressurehPa_a', axis=1)
        X_encoded = X_encoded.drop('Mean VisibilityKm_a', axis=1)
        X_encoded = X_encoded.drop('Mean Wind SpeedKm/h_a', axis=1)
        X_encoded = X_encoded.drop('MeanDew PointC_a', axis=1)
        X_encoded = X_encoded.drop('WindDirDegrees_a', axis=1)
        X_encoded = X_encoded.drop('Dew PointC', axis=1)
        X_encoded = X_encoded.drop('Dew PointC_a', axis=1) 
        X_encoded = X_encoded.drop('CloudCover_a', axis=1)

        dateValues = X_encoded.DateOfDeparture.values
        HighVolumeMonth = []
        yearValues = []
        for d in dateValues:
            m = d.split('-')[1]
            if m==9 or  m==10 or m==11 or m==12 or m==1 or m==2:
                m = 1
            else:
                m = 0
            HighVolumeMonth.append(m)
            y = d.split('-')[0]
            yearValues.append(y)
        X_encoded['DateOfDeparture']=pd.to_datetime(X_encoded['DateOfDeparture'], format='%Y-%m-%d')
        X_encoded['Month'] = HighVolumeMonth
        X_encoded['Month'] = X_encoded['Month'].astype('float64')
        X_encoded['Year'] = yearValues
        X_encoded['Year'] = X_encoded['Year'].astype('float64')
        X_encoded['DateOfDeparture'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("2000-01-01")).days)
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Year'], prefix='y'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('Month', axis=1)
        X_encoded = X_encoded.drop('Year', axis=1)
        X_encoded = X_encoded.drop('WeeksToDeparture', axis=1)
        X_encoded = X_encoded.drop('std_wtd', axis=1)
        scaler = MinMaxScaler()
        columns_to_scale = ['DateOfDeparture', 'Distance', 'Population', 'N_BD', 'MeanDew PointC', 'Mean Wind SpeedKm/h','Mean VisibilityKm',
                    'Mean TemperatureC','Mean Sea Level PressurehPa','Mean Humidity',
                    'CloudCover','WindDirDegrees','Oil_Poffset1','Oil_Poffset3','Oil_Poffset6',
                    'Oil_Poffset12']
        X_encoded[columns_to_scale] = scaler.fit_transform(X_encoded[columns_to_scale]) 
        important= ['DateOfDeparture','Distance',
                    'Mean TemperatureC',
                    'N_BD','Population','d_ATL','d_BOS',
                    'd_CLT','d_DEN','d_DFW','d_DTW','d_EWR','d_IAH','d_JFK','d_LAS',
                    'd_LAX','d_LGA','d_MCO','d_MIA','d_MSP','d_ORD','d_PHL','d_PHX','d_SEA','d_SFO',
                    'a_ATL','a_BOS','a_CLT','a_DEN','a_DFW','a_DTW','a_EWR','a_IAH',
                    'a_JFK','a_LAS','a_LAX','a_LGA','a_MCO','a_MIA','a_MSP',
                    'a_ORD','a_PHL','a_PHX','a_SEA','a_SFO']
        X_encoded=X_encoded[important]
        X_array = X_encoded.values
        return X_array

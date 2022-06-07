###Imports
import pickle
import inflection as inf
import math
import pandas as pd
import numpy  as np
import datetime

class Rossman(object):
    def __init__(self):
        self.home_path = '/media/luanzitto/Install/Luan-PC/Documents/01-Estudos/01-Cursos/01- Comunidade_DS/01-Data_Science_Producao/'
        self.competition_distance_scaler   = pickle.load( open ( self.home_path + 'parameters/competition_distance_scaler.pkl', 'rb' ))
        self.competition_time_month_scaler = pickle.load( open ( self.home_path + 'parameters/competition_time_month_scaler.pkl', 'rb' ))
        self.promo_time_week_scaler        = pickle.load( open ( self.home_path + 'parameters/promo_time_week_scaler.pkl', 'rb' ))
        self.year_scaler                   = pickle.load( open ( self.home_path + 'parameters/year_scaler.pkl', 'rb' ))
        self.store_type_scaler             = pickle.load( open ( self.home_path + 'parameters/store_type_scaler.pkl', 'rb' ))


    def data_cleaning(self, df):

        ###Prepare for rename columns
        old_cols = ['Store', 'DayOfWeek', 'Date','Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        lowercase = lambda x : inf.underscore(x)
        new_cols = list(map(lowercase, old_cols))
        
        #Rename columns
        df.columns = new_cols

        ###Date conversion
        df.date = pd.to_datetime(df.date)

        ###Fill NA values
        #competition_distance
        df.competition_distance = df.competition_distance.apply( lambda x: 200000 if math.isnan( x ) else x )
        #competition_open_since_month  
        df.competition_open_since_month = df.apply(lambda x : x['date'].month if math.isnan(x['competition_open_since_month'])  else x['competition_open_since_month'], axis=1 )       
        #competition_open_since_year
        df.competition_open_since_year = df.apply(lambda x : x['date'].year if math.isnan(x['competition_open_since_year'])  else x['competition_open_since_year'], axis=1 ) 
        #promo2_since_week
        df.promo2_since_week = df.apply(lambda x : x['date'].week if math.isnan(x['promo2_since_week'])  else x['promo2_since_week'], axis=1 ) 
        #promo2_since_year 
        df.promo2_since_year = df.apply(lambda x : x['date'].year if (math.isnan(x['promo2_since_year']))  else x['promo2_since_year'], axis=1 ) 
        #promo_interval        
        df.fillna(0, inplace=True)
        
        ###creation month map
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}        
        df['month_map'] = df['date'].dt.month.map( month_map)
        df['is_promo'] =  df[['month_map', 'promo_interval']].apply(lambda x : 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        ### Change data types
        #competition_open_since_month
        df.competition_open_since_month = df.competition_open_since_month.astype(int)
        df.competition_open_since_year = df.competition_open_since_year.astype(int)        
        #promo2_since_week
        df.promo2_since_week = df.promo2_since_week.astype(int)
        df.promo2_since_year = df.promo2_since_year.astype(int)

        return df

    def feaure_engineering(self, df):
        ###Date Features
        #year
        df['year'] = df.date.dt.year
        #month
        df['month'] = df.date.dt.month
        #day
        df['day'] = df.date.dt.day
        #week_of_year
        df['week_of_year'] = df.date.dt.isocalendar().week
        #year_week
        df['year_week'] = df.date.dt.strftime('%Y-%W')
        # competition_since
        df['competition_since'] =  df[['competition_open_since_month', 'competition_open_since_year']].apply(lambda x : datetime.datetime(year= x['competition_open_since_year'], month= x['competition_open_since_month'], day= 1), axis=1)
        df['competition_time_month'] = ( (df['date'] - df['competition_since']) / 30 ).apply(lambda x : x.days).astype(int)
        # Promo_since
        df['promo_since'] = df['promo2_since_year'].astype(str) + '-' + df['promo2_since_week'].astype(str)
        df['promo_since'] = df['promo_since'].apply(lambda x : datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df['promo_time_week'] = ( (df['date'] - df['promo_since'] ) / 7 ).apply(lambda x : x.days).astype(int)
        # state_holiday
        df['state_holiday'] = df['state_holiday'].apply(lambda x : 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        ###Assortment feature
        df['assortment'] = df['assortment'].apply(lambda x : 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended' if x == 'c' else 'invalid')

        ### Line filtering
        df = df.loc[df['open'] != 0]

        ### Colunms filterign
        cols_drop = ['open','promo_interval', 'month_map']
        df = df.drop(cols_drop, axis=1)

        return df

    def data_preparation(self, df):
        ###Rescaling
        #competition_distance
        df['competition_distance'] =  self.competition_distance_scaler.transform(df[['competition_distance']].values)  
        #competition_time_month     
        df['competition_time_month'] =self.competition_time_month_scaler.transform(df[['competition_time_month']].values)
        #promo_time_week
        df['promo_time_week'] =self.promo_time_week_scaler.transform(df[['promo_time_week']].values)
        #year
        df['year'] =self.year_scaler.transform(df[['year']].values)

        ###Encoding
        #state_holiday | One hot encoding
        df = pd.get_dummies(df, prefix=['state_holiday'], columns=['state_holiday'])
        #store_type    | Label encoding
        df['store_type'] = self.store_type_scaler.transform(df['store_type'])
        #assortment    | Ordinal encoding 
        assortment_dict = {'basic':1, 'extra':2, 'extended':3}
        df.assortment = df.assortment.map(assortment_dict)
       
        #Nature Transformation
        #day_of_week
        df['day_of_week_sin'] = df['day_of_week'].apply(lambda x: np.sin( x* (2. * np.pi/7 ) ) )
        df['day_of_week_cos'] = df['day_of_week'].apply(lambda x: np.cos( x* (2. * np.pi/7 ) ) )
        #month
        df['month_sin'] = df['month'].apply(lambda x: np.sin( x* (2. * np.pi/12 ) ) )
        df['month_cos'] = df['month'].apply(lambda x: np.cos( x* (2. * np.pi/12 ) ) )
        #day
        df['day_sin'] = df['day'].apply(lambda x: np.sin( x* (2. * np.pi/30 ) ) )
        df['day_cos'] = df['day'].apply(lambda x: np.cos( x* (2. * np.pi/30 ) ) )
        #week_of_year
        df['weekofyear_sin'] = df['week_of_year'].apply(lambda x: np.sin( x* (2. * np.pi/52 ) ) )
        df['weekofyear_cos'] = df['week_of_year'].apply(lambda x: np.cos( x* (2. * np.pi/52 ) ) )

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 
                                'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 
                                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekofyear_sin', 'weekofyear_cos'
                               ]

        return df[cols_selected]  

    def get_prediction(self, model, original_data, test_data):
        #Prediction
        pred = model.predict( test_data )

        #Join pred into the original data
        original_data['prediction'] =  np.expm1(pred)   

        return original_data.to_json(orient='records', date_format='iso')
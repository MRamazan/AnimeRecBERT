from .base import AbstractDataset
import struct
import pandas as pd
import numpy as np
from datetime import date
import time
from pathlib import Path


class AR54MDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'AnimeRatings54M'

    @classmethod
    def url(cls):
        return 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'movies.dat',
                'ratings.dat',
                'ratings.bin'
               ]

    def load_ratings_df_datfile(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('ratings.dat')
        df = pd.read_csv(file_path, sep='::', header=None)
        
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        print(np.max(df["sid"].to_numpy()))
        return df


    def load_ratings_df(self):
     """
     Load ratings from numpy file and return DataFrame in the same format as load_ratings_df_datfile
    
     Args:
        folder_path: Path to the folder containing ratings.npy file
                    If None, uses self._get_rawdata_folder_path()
    
     Returns:
        pandas.DataFrame: DataFrame with columns ['uid', 'sid', 'rating', 'timestamp']
     """
    
     folder_path = self._get_rawdata_folder_path()
    
     file_path = Path(folder_path).joinpath('ratings.npy')
    
     # Load numpy array
     # ratings_array shape: (n_ratings, 3) - columns: [user_id, movie_id, rating]
     ratings_array = np.load(file_path)
    
     # Create DataFrame from numpy array
     df = pd.DataFrame(ratings_array, columns=['uid', 'sid', 'rating'])
    
     # Add timestamp column (set to default value as in original converter)
     # Using the same default timestamp as in the converter: 978300760
     df['timestamp'] = 978300760
    
     # Reorder columns to match the original format
     df = df[['uid', 'sid', 'rating', 'timestamp']]
    
     print(f"Loaded {len(df):,} ratings from numpy file")
     print(f"Max sid: {np.max(df['sid'].to_numpy())}")
     print(f"Max uid: {np.max(df['uid'].to_numpy())}")
     print(f"Rating range: {np.min(df['rating'].to_numpy())} - {np.max(df['rating'].to_numpy())}")
    
     return df

    


    




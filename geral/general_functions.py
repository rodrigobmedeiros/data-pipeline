import pandas as pd


def read_dataframe(file_dir: str, filename: str) -> pd.DataFrame:
    """
    function responsible to read dataframes into correct format to be used by the pipeline.
    """

    df = pd.read_csv(''.join([file_dir, filename]))
    df = format_dataframe(df)

    return df

def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    function responsible for format dataframe 
    """

    df['data'] = pd.to_datetime(df['data'])
    df.set_index('data', inplace=True)

    return df


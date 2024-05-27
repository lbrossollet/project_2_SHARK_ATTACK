import pandas as pd
import numpy as np


def clean_year_column_f(df, column_name, valid_start_year=1900):
    """
    Cleans and standardizes a year column in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the year column.
    column_name (str): The name of the year column to clean.
    valid_start_year (int): The start year for valid years (default is 1900).
    
    Returns:
    pd.DataFrame: A DataFrame with the cleaned year column.
    """
    # step 1:Replace 0.0 with NaN to handle them later
    df[column_name] = df[column_name].replace(0.0, np.nan)

    # step 2 :Drop rows with NaN values in the specified year column
    df = df.dropna(subset=[column_name])

    # step 3 Convert the year column to integer type
    df[column_name] = df[column_name].astype(int)

    # step 4:Remove or filter invalid years
    df = df[df[column_name] >= valid_start_year]

    return df


def extract_month_f(df, pdf_column='pdf'):
    """
    Extracts the date and month from the pdf column in a DataFrame,
    and moves the 'month' column to the second position.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    pdf_column (str): The name of the column containing the PDF file names.

    Returns:
    pd.DataFrame: The modified DataFrame with an additional 'month' column moved to the second position.
    """
    # Convert the specified column to string type
    df[pdf_column] = df[pdf_column].astype(str)

    # Extract the date2 part from the specified column
    df['date2'] = df[pdf_column].str.extract(r'(\d{4}\.\d{2}\.\d{2})')

    # Convert the extracted date2 string to a datetime object
    df['date2'] = pd.to_datetime(df['date2'], format='%Y.%m.%d', errors='coerce')

    # Extract the month from the datetime object
    df['month'] = df['date2'].dt.month

    # Drop the intermediate 'date2' column
    df.drop(columns=['date2'], inplace=True)

    # Move the 'month' column to the second position
    cols = df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('month')))
    df = df[cols]

    return df

def clean_sex_column_f(df, column_name='sex'):
    """
    Cleans the specified column by converting all string values to lowercase
    and removing any spaces.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    column_name (str): The name of the column to clean.

    Returns:
    pd.DataFrame: The modified DataFrame with the cleaned column.
    """
    df[column_name] = df[column_name].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df[column_name] = df[column_name].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
    return df

def extract_month_from_date_f(df, date_column='date', month_column='month'):
    """
    Extracts the month from a specified Date column where the month column is NaN,
    and updates the month column with the numeric month value.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    date_column (str): The name of the column containing the date.
    month_column (str): The name of the column containing the month.

    Returns:
    pd.DataFrame: The modified DataFrame with the updated month column.
    """
    # Create a mapping from month abbreviation to month number
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Extract the month part from the date column where the month column is NaN
    mask = df[month_column].isna()
    
    # Extract the month abbreviation using slicing
    extracted_month_abbr = df.loc[mask, date_column].str[-8:-5]
    
    # Map the extracted month abbreviation to the corresponding month number
    df.loc[mask, month_column] = extracted_month_abbr.map(month_mapping)
    
    # Ensure the month column contains numeric values
    df[month_column] = pd.to_numeric(df[month_column], errors='coerce')

    return df

    
def normalize_case_f(country):
    if pd.isna(country):
        return country
    return country.title()

def standardize_names_f(country):
    corrections = {
        'Usa': 'United States',
        'Uk': 'United Kingdom',
        'England': 'United Kingdom',
        'Ceylon (Sri Lanka)': 'Sri Lanka',
        'Trinidad & Tobago': 'Trinidad and Tobago',
        'Trinidad': 'Trinidad and Tobago',
        'St. Martin': 'Saint Martin',
        'St. Maartin': 'Saint Martin',
        'St Martin': 'Saint Martin',
        'St Kitts / Nevis': 'Saint Kitts and Nevis',
        'United Arab Emirates (Uae)': 'United Arab Emirates',
        'Reunion Island': 'Reunion',
        'St Helena, British Overseas Territory': 'Saint Helena',
        'Turks And Caicos': 'Turks and Caicos Islands',
        'Turks & Caicos': 'Turks and Caicos Islands',
        'Columbia': 'Colombia',
        'Papua New Guinea': 'Papua New Guinea',
        'British Overseas Territory': 'British Overseas Territories',
        'Palestinian Territories': 'Palestine',
        'New Caledonia': 'New Caledonia',
        'Northern Arabian Sea': 'Arabian Sea',
        'Andaman Islands': 'Andaman and Nicobar Islands',
        'Equatorial Guinea / Cameroon': 'Equatorial Guinea',
        'Mediterranean Sea': 'Mediterranean',
        'Red Sea?': 'Red Sea',
        'Asia?': 'Asia',
        'Egypt / Israel': 'Egypt',
        'Between Portugal & India': 'India',
        'Africa': 'African continent',
        'Coast Of Africa': 'African coast',
        'Tasman Sea': 'Tasman Sea region',
        'Caribbean Sea': 'Caribbean',
        'Atlantic Ocean': 'Atlantic',
        'Okinawa': 'Japan',
        'Roatan': 'Honduras',
        'Greenland': 'Denmark',
        'St Kitts': 'Saint Kitts and Nevis',
        'St Helena': 'Saint Helena',
        'Bahrein': 'Bahrain',
        'Diego Garcia': 'British Indian Ocean Territory',
        'Guam': 'United States',
        'Hong Kong': 'China',
        'Korea': 'South Korea'
    }
    return corrections.get(country, country)

def filter_non_countries_f(country):
    non_countries = {'Tasman Sea region','Mediterranean','Indian Ocean?','Atlantic Ocean', 'Coral Sea', 'Tasman Sea', 'Mediterranean Sea', 'Caribbean Sea', 'Pacific Ocean', 'Indian Ocean', 'Red Sea', 'Gulf of Aden', 'Northern Arabian Sea', 'Asia', 'African continent', 'African coast'}
    if country in non_countries:
        return np.nan
    return country


def normalize_activity_f(activity):
    if pd.isna(activity):
        return activity
      
    normalized = activity.lower().replace(' ', '')

    activity_corrections = {
        'scubadiving': 'scuba diving',
        'bodyboarding': 'body boarding',
        'bodysurfing': 'body surfing',
        'boogieboarding': 'boogie boarding',
        'freediving': 'free diving',
        'stand-uppaddleboarding': 'stand-up paddleboarding',
        'kayakfishing': 'kayak fishing',
        'fishingforsharks': 'shark fishing',
        'surfing': 'surfing',
        'swimming': 'swimming',
        'fishing': 'fishing',
        'spearfishing': 'spearfishing',
        'wading': 'wading',
        'bathing': 'bathing',
        'diving': 'diving',
        'snorkeling': 'snorkeling',
        'standing': 'standing',
        'treadingwater': 'treading water',
        'felloverboard': 'fell overboard',
        'pearldiving': 'pearl diving',
        'windsurfing': 'windsurfing',
        'walking': 'walking',
        'canoeing': 'canoeing',
        'floating': 'floating',
        'sharkfishing': 'shark fishing',
        'surffishing': 'surf fishing',
        'playing': 'playing',
        'surf-skiing': 'surf skiing',
        'rowing': 'rowing',
        'surf-skiing': 'surf skiing',
        'paddleboarding': 'paddle boarding',
        'sponge diving': 'sponge diving',
        'divingfortrochus': 'diving for trochus',
        'surfskiiing': 'surf skiing',
        'seadisaster': 'sea disaster',
        'skindiving': 'free diving',
     }
    
    return activity_corrections.get(normalized, normalized)


def format_age_f (age) : 
    if isinstance (age,str) :
        age = age.strip().lower()
        if 'a min' in age :
            return np.nan
        if '?' in age :
            return np.nan
        if',' and '&' in age : 
            return np.nan
        if '&' in age : 
            return int(age.split('&')[0].strip())
        if 'or' in age : 
            return int(age.split('or')[0].strip())
        if 'to' in age : 
            return int(age.split('to')[0].strip())
        if 's' in age and age.replace('s','').isdigit():
            return int(age.replace('s',''))
        if "'s" in age and age.replace("'s" ,'').isdigit():
            return int(age.replace("'s" ,''))
        if '½' in age :
            return int(age.split('½')[0].strip())       
        if age.isdigit():
            return int(age)
        else : 
            return np.nan
    elif isinstance (age,int):
        return age 
    else :
        return np.nan



def fatal_f (x):
    if pd.isna(x):
        return 'UNKNOWN'
    if isinstance(x, int):           
        return 'UNKNOWN'  
    x = str(x.strip().lower())
    if x == 'y' or x == 'f':           
        return 'Yes'     
    elif x == 'n':
        return 'No'
    else:
        return 'UNKNOWN'

def age_group_f(age):
    if age < 13:
        return 'kids'
    elif 13 <= age <= 17:
        return 'teen'
    elif 18 <= age <= 29:
        return 'young adult'
    elif 30 <= age <= 49:
        return 'adult'
    elif age >= 50:
        return 'senior'
    else:
        return 'unknown'


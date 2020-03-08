import pandas as pd


### LOAD DATA ###
def load_raw_data(base_path):
    # Load data
    calendar = pd.read_csv(base_path + 'calendar.csv')
    sell_prices = pd.read_csv(base_path + 'sell_prices.csv')
    sales = pd.read_csv(base_path + 'sales_train_validation.csv')
    return calendar, sell_prices, sales

### CLEAN DATA ###
def clean_data(df:object, id_vars:list, time_var:str, target:str):
    # Lowercase columns
    df.columns = df.columns.str.lower()
    # Tidy Data
    df = df.melt(id_vars=id_vars, var_name=time_var, value_name=target)
    # Convert time var
    return df

### CREATE TRAIN AND VALIDATION SET ###
def train_val_set(df:object, time_var:str, window:int, shift:int, start:int, target:str):
    # Cut lower bound
    val = df[df[time_var] >= df[time_var].max() - window - shift]
    train = df[df[time_var] >= start - shift]
    # Cut upper bound
    train = train[train[time_var] < train[time_var].max() - window - shift]
    val = val[val[time_var] <= val[time_var].max() - shift]

    # Info about train and test set
    train_min, train_max = train[time_var].min(), train[time_var].max()
    val_min, val_max = val[time_var].min(), val[time_var].max()
    
    print(f'Training set from day {train_min} to {train_max} with {len(train)} observations.')
    print(f'Validation set from day {val_min} to {val_max} with {len(val)} observations.')

    # Split x and y
    train_x, train_y = train.drop(target, axis=1), train[target]
    val_x, val_y = val.drop(target, axis=1), val[target]

    return train_x, train_y, val_x, val_y

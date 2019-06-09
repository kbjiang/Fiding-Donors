import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# numerical feature processing
def ft_num_engineer(data, ft_num):
    
    #import numpy as np
    # logrithmic transform on 'capital-gain' and 'capital-loss'

    data['capital-gain']=np.log(data['capital-gain'] + 1)
    data['capital-loss']=np.log(data['capital-loss'] + 1)

    # scaling the features
    # scaling works on multiple featurs simultaneously
    scaler = MinMaxScaler()
    data[ft_num] = scaler.fit_transform(data[ft_num])
    
    return data
 
# categorical feature processing
def ft_cat_eda(data, workclass, occupation, marital, relationship, race):

    # workclass
    workclass_dict = {' Without-pay':'without-pay',' State-gov':'employee', ' Federal-gov':'employee', ' Local-gov':'employee', \
                      ' Private':'employee', ' Self-emp-not-inc':'employee', ' Self-emp-inc':'owner'}
    # occupation
    occupation_income = pd.Series([0, 1, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1]).map({0:'low', 1:'mid', 2:'high'})
    occupation_dict = dict(zip(sorted(data['occupation'].unique()), occupation_income))
    # marital-status
    marital_group = pd.Series([1 if x in [' Married-civ-spouse', ' Married-AF-spouse'] else 0 for x in data['marital-status'].unique()]).map({0:'single', 1:'couple'})
    marital_dict = dict(zip(data['marital-status'].unique(), marital_group))
    # relationship
    relationship_group = pd.Series([1 if x in [' Husband', ' Wife'] else 0 for x in data['relationship'].unique()]).map({0:'single', 1:'couple'})
    relationship_dict = dict(zip(data['relationship'].unique(), relationship_group))
    # race
    race_dict = {' White': 'high', ' Asian-Pac-Islander': 'high', ' Black':'low', ' Amer-Indian-Eskimo':'low', ' Other':'low'}
    
    # replacement
    filter = np.array([workclass, occupation, marital, relationship, race])
    fts = np.array(['workclass', 'occupation', 'marital-status', 'relationship', 'race'])
    dicts = np.array([workclass_dict, occupation_dict, marital_dict, relationship_dict, race_dict])
    
    replace_dict = dict(zip(fts[filter], dicts[filter]))

    data = data.replace(replace_dict)
    
    return data

# put all together
def ft_engineer(data, capital=False, workclass=False, occupation=False, marital=False, \
                    relationship=False, race=False, drop_native_country=False):

    # numerical engineering
    ft_num = data.select_dtypes(include=['int64','float64']).columns.values    
    data = ft_num_engineer(data, ft_num)
      
    # should we binarize 'capital'
    if capital:
        data['capital-gain']= data['capital-gain'].apply(lambda x: 'no' if x==0 else 'yes')
        data['capital-loss']= data['capital-loss'].apply(lambda x: 'no' if x==0 else 'yes')
    
    # EDA suggested update
    data = ft_cat_eda(data, workclass, occupation, marital, relationship, race)
    
    # should we drop 'native-country'
    if drop_native_country:
        data.drop('native-country', axis=1, inplace=True)

    # target and get_dummies
    target = np.array(data['income'] != '<=50K').astype(int)
    
    #data.drop(['education_level', 'income'], axis=1, inplace=True)
    data.drop(['income'], axis=1, inplace=True)
    data = pd.get_dummies(data)
    
    return data, target

if __name__ == '__main__':
    print('No direct calling of this module.')
    
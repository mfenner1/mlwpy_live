import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.model_selection as skms
from sklearn import metrics

def sk_table_to_tidy(train_scores, # y values
                     test_scores,  # y values
                     eval_points,  # x values
                     eval_label,   # x column name
                     score_label): # y column name
    assert train_scores.shape[1] == test_scores.shape[1]
    num_folds = train_scores.shape[1]

    # construct row-by-row labels (a pct, train/test, fold)
    # and place in table
    labels = easy_combo(eval_points, 
                        [0,1], # surrogates for train/test 
                        np.arange(num_folds))
    df = pd.DataFrame.from_records(labels)
    df.columns = [eval_label, 'Set', 'Fold']
    df.Set = df.Set.replace({0:'Train', 1:'Test'})

    # construct the result (score) column
    score = np.concatenate([train_scores.flatten(), 
                            test_scores.flatten()], axis=0)    
    df[score_label] = score
    return df

def easy_combo(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays),
                     axis=-1)
              .reshape(-1, ndim))

def rms_error(actual, predicted):
    mse = metrics.mean_squared_error(actual, predicted)
    return np.sqrt(mse)

rmse_scorer = metrics.make_scorer(rms_error)
rmse = rmse_scorer

def make_learning_curve(model, model_name, ftrs, tgt, ax=None):
    train_sizes = np.linspace(.1, 1.0, 10)
    lcr = skms.learning_curve(model, ftrs, tgt,
                              cv=5, train_sizes=train_sizes,
                              scoring=rmse_scorer)
    (train_N, train_scores, test_scores) = lcr
    
    neat_sizes = (train_sizes*100).astype(np.int)
    tidy_df = sk_table_to_tidy(train_scores, test_scores,
                               neat_sizes, 'Percent',
                               'RMSE')

    ax = sns.lineplot(x='Percent', y='RMSE', hue='Set', data=tidy_df,
                      legend='brief', ax=ax)
    ax.set(title="Learning Curve for " + model_name,
           xlabel="Percent of Data used for Training",
           ylabel="RMSE",
           ylim=(.0, 1.3))
    return ax

def make_complexity_curve(model, model_name,
                          param_name, param_range,
                          ftrs, tgt, ax=None):
    results = skms.validation_curve(model, ftrs, tgt,
                                    param_name=param_name,
                                    param_range=param_range,
                                    cv=5,
                                    scoring=rmse_scorer)
    (train_scores, test_scores) = results

    tidy_df = sk_table_to_tidy(train_scores, test_scores,
                               param_range, 'k', 'RMSE')
    ax = sns.lineplot(x='k', y='RMSE', hue='Set', data=tidy_df,
                      legend='brief', ax=ax)
    ax.set(title='5-fold CV Performance for ' + model_name,
           xlabel=param_name,
           xticks=param_range,
           ylim=(.0, 1.3),
           ylabel='RMSE')
    return ax



def manage_ames_ordinal(ames_df):
    # note:  OverallCond OverallQual are already numerically coded
    qualities = ['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex']
    ordinal_qualities = ["ExterQual",  "ExterCond", 
                         "BsmtQual",   "BsmtCond",
                         "HeatingQC",  "KitchenQual","FireplaceQu", 
                         "GarageQual", "GarageCond", 
                         "PoolQC"]
    quality_replacers = {sym:val for val,sym in enumerate(qualities)}
    quality_mapping   = {oq:quality_replacers for oq in ordinal_qualities}

    bsmtfin_type = ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']
    ordinal_values = \
        {"LotShape"    : ["IR3", "IR2", "IR1", "Reg"],
         "Utilities"   : ['ELO', 'NoSeWa', 'NoSewr', 'AllPub'],
         "LandSlope"   : ['Sev', 'Mod', 'Gtl'],
         "BsmtExposure": ['NA', 'No', 'Mn', 'Av', 'Gd'],
         "BsmtFinType1": bsmtfin_type,
         "BsmtFinType2": bsmtfin_type,
         "Electrical"  : ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'],
         "Functional"  : ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
         "GarageFinish": ['NA', 'Unf', 'RFn', 'Fin'],
         "PavedDrive"  : ['N', 'P', 'Y'],
         "Fence"       : ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"]}
    other_mapping = {name:{s:v for v,s in enumerate(vals)}
                       for name, vals in ordinal_values.items()}
    
    # no dict union until python 3.9
    ordinal_mapping = {**quality_mapping, **other_mapping}
    ordinal_names = list(ordinal_mapping) # keys
    # make sure we didn't make any typos in column names
    assert all(k in ames_df.columns for k in ordinal_mapping)
    
    ames_df = ames_df.replace(ordinal_mapping)
    ames_df[ordinal_names] = ames_df[ordinal_names].fillna(0)
    
    return ames_df

def fill_with_mean(grp):  
    return grp.fillna(grp.mean())

def manage_ames_nans(ames_df):
    # deal with NaNs
    ames_df['Alley'] = ~ames_df['Alley'].isna()
    ames_df['LotFrontage'] = (ames_df[['LotConfig', 'LotFrontage']].groupby('LotConfig')
                                                                   .transform(fill_with_mean))
    ames_df['MasVnrArea'].fillna(0, inplace=True)

    # just dropping these 
    ames_df = ames_df.drop(columns="GarageYrBlt")
    
    # remove some outliers (as recommended by documentation)
    too_big = ames_df['GrLivArea'] > 4000
    ames_df = ames_df[~too_big] # .copy()
    return ames_df

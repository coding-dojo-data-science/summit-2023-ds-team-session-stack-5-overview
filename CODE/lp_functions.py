import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
## Using pd.set_option to display more columns
pd.set_option('display.max_columns',100)

# from lp_styles import *
# ## Customization Options
# plt.style.use(['fivethirtyeight','seaborn-talk'])
# mpl.rcParams['figure.facecolor']='white'

## additional required imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import metrics

SEED = 321
np.random.seed(SEED)

def show_code(function):
    """Display the source code of a funciton or module.
    Uses the inspect and IPython modules to display Markdown with Python Syntax. 
    Args:
        function (function or module object): Pass the function/module to show. 
                                              Use function name, no parentheses.
        
    Example Use:
    ## Example with Function
    >> import CODE.lp_functions as lp
    >> import scipy.stats as stats
    >> lp.show_code(stats.ttest_ind)  
    """
    
    import inspect 
    from IPython.display import display,Markdown
    
    code = inspect.getsource(function)
    md_txt = f"```python\n{code}\n```"
    return display(Markdown(md_txt))



def get_importances(model, feature_names=None,name='Feature Importance',
                   sort=False, ascending=True):
    """Extracts and returns model.feature_importances_ 
    
    Args:
        model (sklearn estimator): a fit model with .feature_importances_
        feature_names (list/array): the names of the features. Default=None.
                                    If None, extract feature names from model
        name (str): name for the panda's Series. Default is 'Feature Importance'
        sort (bool): controls if importances are sorted by value. Default=False.
        ascending (bool): ascending argument for .sort_values(ascending= ___ )
                            Only used if sort===True.
                            
    Returns:
        Pandas Series with Feature Importances
        """
    import pandas as pd
    
    ## checking for feature names
    if feature_names is None:
        feature_names = model.feature_names_in_
        
    ## Saving the feature importances
    importances = pd.Series(model.feature_importances_, index= feature_names,
                           name=name)
    
    # sort importances
    if sort == True:
        importances = importances.sort_values(ascending=ascending)
        
    return importances



def plot_importance(importances, top_n=None,  figsize=(8,6)):
    # sorting with asc=false for correct order of bars
    
    if top_n==None:
        ## sort all features and set title
        plot_importances = importances.sort_values()
        title = "All Features - Ranked by Importance"

    else:
        ## sort features and keep top_n and set title
        plot_importances = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"

    ## plotting top N importances
    ax = plot_importances.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Importance', 
           ylabel='Feature Names', 
           title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax


# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def evaluate_regression(model, X_train,y_train, X_test, y_test): 
    """Evaluates a scikit learn regression model using r-squared and RMSE"""
    from sklearn import metrics
    
    ## Training Data
    y_pred_train = model.predict(X_train)
    r2_train = metrics.r2_score(y_train, y_pred_train)
    rmse_train = metrics.mean_squared_error(y_train, y_pred_train, 
                                            squared=False)
    
    print(f"Training Data:\tR^2= {r2_train:.2f}\tRMSE= {rmse_train:.2f}")
        
    
    ## Test Data
    y_pred_test = model.predict(X_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)
    rmse_test = metrics.mean_squared_error(y_test, y_pred_test, 
                                            squared=False)
    
    print(f"Test Data:\tR^2= {r2_test:.2f}\tRMSE= {rmse_test:.2f}")
    

    
### COEFFICIENTS 
# def get_coeffs_linreg(lin_reg, feature_names = None, sort=True,ascending=True,
#                      name='LinearRegression Coefficients'):
#     if feature_names is None:
#         feature_names = lin_reg.feature_names_in_

#     ## Saving the coefficients
#     coeffs = pd.Series(lin_reg.coef_, index= feature_names)
#     coeffs['intercept'] = lin_reg.intercept_
    
#     if sort==True:
#         coeffs = coeffs.sort_values(ascending=ascending)
    
#     return coeffs
def get_coeffs_linreg(lin_reg, feature_names = None, intercept=False,
                      sort=True,ascending=True,
                     name='LinearRegression Coefficients'):
    if feature_names is None:
        feature_names = lin_reg.feature_names_in_

    ## Saving the coefficients
    coeffs = pd.Series(lin_reg.coef_, index= feature_names)
    
    if intercept == True:
        coeffs['intercept'] = lin_reg.intercept_
    
    if sort==True:
        coeffs = coeffs.sort_values(ascending=ascending)
    
    return coeffs

    
def get_coeffs(model, feature_names=None,name='Coefficients',
                   sort=False, ascending=True):
    import warnings
    warnings.warn('Function has been replaced with: get_coeffs_linreg and get_coeffs_logreg')
    
    ## checking for feature names
    if feature_names == None:
        feature_names = model.feature_names_in_

    ## Saving the coefficients
    coeffs = pd.Series(model.coef_, index= feature_names)
    coeffs['intercept'] = model.intercept_
    coeffs.name = name

    # sort importances
    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)
        
    return coeffs


# def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
#                       name='LogReg Coefficients', class_index=0, 
#                       include_intercept=False, as_odds=False):
#     if feature_names is None:
#         feature_names = logreg.feature_names_in_
        
    
#     ## Saving the coefficients
#     coeffs = pd.Series(logreg.coef_[class_index],
#                        index= feature_names, name=name)
    
#     if include_intercept:
#         # use .loc to add the intercept to the series
#         coeffs.loc['intercept'] = logreg.intercept_[class_index]
        
#     if as_odds==True:
#         coeffs = np.exp(coeffs)

#     if sort == True:
#         coeffs = coeffs.sort_values(ascending=ascending)
    
        
#     return coeffs
def get_coeffs_logreg(logreg, feature_names = None, sort=True,ascending=True,
                      name='LogReg Coefficients', class_index=0):
    
    if feature_names is None:
        feature_names = logreg.feature_names_in_
        

    ## Saving the coefficients
    coeffs = pd.Series(logreg.coef_[class_index],
                       index= feature_names, name=name)
    
    # use .loc to add the intercept to the series
    coeffs.loc['intercept'] = logreg.intercept_[class_index]

    if sort == True:
        coeffs = coeffs.sort_values(ascending=ascending)
        
    return coeffs


# def plot_coeffs(coeffs, top_n=None,  figsize=(8,6)):

#     if top_n==None:
#         ## sort all features and set title
#         plot_vals = coeffs.sort_values()
#         title = "All Coefficients - Ranked by Magnitude"

#     else:
#         ## rank the coeffs and select the top_n
#         coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
#         top_n_features = coeff_rank.head(top_n)

#         plot_vals = coeffs.loc[top_n_features.index].sort_values()
#         ## sort features and keep top_n and set title
#         title = f"Top {top_n} Largest Coefficients"

#     ## plotting top N importances
#     ax = plot_vals.plot(kind='barh', figsize=figsize)
#     ax.set(xlabel='Coefficient', 
#            ylabel='Feature Names', 
#            title=title)
#     ax.axvline(0, color='k')
    
#     ## return ax in case want to continue to update/modify figure
#     return ax
def plot_coeffs(coeffs, top_n=None,  figsize=(4,5), intercept=False,
                annotate=False, ha='left', va='center', size=12, xytext=(4,0),
                textcoords='offset points'):
    
    if intercept==False:
        coeffs = coeffs.drop('intercept')
        
    if top_n==None:
        ## sort all features and set title
        plot_vals = coeffs#.sort_values()
        title = "All Coefficients - Ranked by Magnitude"

    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)

        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        ## sort features and keep top_n and set title
        title = f"Top {top_n} Largest Coefficients"

    ## plotting top N importances
    ax = plot_vals.plot(kind='barh', figsize=figsize)
    ax.set(xlabel='Coefficient', 
           ylabel='Feature Names', 
           title=title)
    ax.axvline(0, color='k')
    
    if annotate==True:
        annotate_hbars(ax, ha=ha,va=va,size=size,xytext=xytext,
                       textcoords=textcoords)
    ## return ax in case want to continue to update/modify figure
    return ax


def annotate_hbars(ax, ha='left',va='center',size=12,  xytext=(4,0),
                  textcoords='offset points'):
    for bar in ax.patches:
        
        ## get the value to annotate
        val = bar.get_width()

        if val<0:
            x=0
        else:
            x=val


        ## calculate center of bar
        bar_ax = bar.get_y() + bar.get_height()/2

        # ha and va stand for the horizontal and vertical alignment
        ax.annotate(f"{val:.3f}", (x,bar_ax),ha=ha,va=va,size=size,
                    xytext=xytext, textcoords=textcoords)

        
        

def get_importances(model, feature_names=None,name='Feature Importance',
                   sort=False, ascending=True):
    """Extracts and returns model.feature_importances_ 
    
    Args:
        model (sklearn estimator): a fit model with .feature_importances_
        feature_names (list/array): the names of the features. Default=None.
                                    If None, extract feature names from model
        name (str): name for the panda's Series. Default is 'Feature Importance'
        sort (bool): controls if importances are sorted by value. Default=False.
        ascending (bool): ascending argument for .sort_values(ascending= ___ )
                            Only used if sort===True.
                            
    Returns:
        Pandas Series with Feature Importances
        """
    
    ## checking for feature names
    if feature_names == None:
        feature_names = model.feature_names_in_
        
    ## Saving the feature importances
    importances = pd.Series(model.feature_importances_, index= feature_names,
                           name=name)

    # sort importances
    if sort == True:
        importances = importances.sort_values(ascending=ascending)
        
    return importances




def plot_importance_color(importances, top_n=None,  figsize=(8,6), 
                          color_dict=None):
    """Plots series of feature importances
    
    Args:
        importances (pands Series): importance values to plot
        top_n (int): The # of features to display (Default=None). 
                        If None, display all.
                        otherwise display top_n most important
                        
        figsize (tuple): figsize tuple for .plot
        color_dict (dict): dict with index values as keys with color to use as vals
                            Uses series.index.map(color_dict).
                            
    Returns:
        Axis: matplotlib axis
        """
    # sorting with asc=false for correct order of bars
    if top_n==None:
        ## sort all features and set title
        plot_vals = importances.sort_values()
        title = "All Features - Ranked by Importance"

    else:
        ## sort features and keep top_n and set title
        plot_vals = importances.sort_values().tail(top_n)
        title = f"Top {top_n} Most Important Features"


    ## create plot with colors, if provided
    if color_dict is not None:
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        ax = plot_vals.plot(kind='barh', figsize=figsize, color=colors)
        
    else:
        ## create plot without colors, if not provided
        ax = plot_vals.plot(kind='barh', figsize=figsize)
        
    # set titles and axis labels
    ax.set(xlabel='Importance', 
           ylabel='Feature Names', 
           title=title)
    
    ## return ax in case want to continue to update/modify figure
    return ax



def get_color_dict(importances, color_rest='#006ba4' , color_top='green', 
                   top_n=7):
    """Constructs a color dictionary where the index of the top_n values will be 
    colored with color_top and the rest will be colored color_rest"""
    ## color -coding top 5 bars
    highlight_feats = importances.sort_values(ascending=True).tail(top_n).index
    colors_dict = {col: color_top if col in highlight_feats else color_rest for col in importances.index}
    return colors_dict
    
# def get_report(model,X_test,y_test,as_df=False,label="TEST DATA"):
#     """Get classification report from sklearn and converts to DataFrame"""
#     ## Get Preds and report
#     y_hat_test = model.predict(X_test)
#     scores = metrics.classification_report(y_test, y_hat_test,
#                                           output_dict=as_df) 
#     ## convert to df if as_df
#     if as_df:
#         report = pd.DataFrame(scores).T.round(2)
#         report.iloc[2,[0,1,3]] = ''
#         return report
#     else:
#         header="\tCLASSIFICATION REPORT"
#         if len(label)>0:
#             header += f" - {label}"
#         dashes='---'*20
#         print(f"{dashes}\n{header}\n{dashes}")
#         print(scores)
        
        
        
def evaluate_classification(model, X_train,y_train,X_test,y_test,
                            normalize='true',cmap='Blues', figsize=(10,5)):
    header="\tCLASSIFICATION REPORT"
    dashes='--'*40
    print(f"{dashes}\n{header}\n{dashes}")

    ## training data
    print(f"[i] Training Data:")
    y_pred_train = model.predict(X_train)
    report_train = metrics.classification_report(y_train, y_pred_train)
    print(report_train)

    fig,ax = plt.subplots(figsize=figsize,ncols=2)
    metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
                                                  normalize=normalize, 
                                                  cmap=cmap,ax=ax[0])
    metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].grid()
    fig.tight_layout()
    plt.show()
    
    print(dashes)
    ## training data
    print(f"[i] Test Data:")
    y_pred_test = model.predict(X_test)
    report_test = metrics.classification_report(y_test, y_pred_test)
    print(report_test)

    fig,ax = plt.subplots(figsize=figsize,ncols=2)
    metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
                                                  normalize=normalize, 
                                                  cmap=cmap, ax=ax[0])
    metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
    ax[1].plot([0,1],[0,1],ls=':')
    ax[1].grid()
    fig.tight_layout()
    plt.show()
    
    
def get_colors_gt_lt(coeffs, threshold=1, color_lt ='darkred', 
                    color_gt='forestgreen',color_else='gray'):
    """Creates a dictionary of features:colors based on if value is > or < threshold"""
    
    colors_dict = {}

    for i in coeffs.index:

        rounded_coeff = np.round( coeffs.loc[i],3)

        if rounded_coeff < threshold:
            color = color_lt

        elif rounded_coeff > threshold:
            color = color_gt

        else:
            color=color_else

        colors_dict[i] = color
        
    return colors_dict


def plot_coeffs_color(coeffs, top_n=None,  figsize=(8,6), intercept=False,
                      legend_loc='best', threshold=None, 
                      color_lt='darkred', color_gt='forestgreen',
                      color_else='gray', label_thresh='Equally Likely',
                      label_gt='More Likely', label_lt='Less Likely',
                      plot_kws = {}):
    """Plots series of coefficients
    
    Args:
        ceoffs (pands Series): importance values to plot
        top_n (int): The # of features to display (Default=None). 
                        If None, display all.
                        otherwise display top_n most important
                        
        figsize (tuple): figsize tuple for .plot
        color_dict (dict): dict with index values as keys with color to use as vals
                            Uses series.index.map(color_dict).
        plot_kws (dict): additional keyword args accepted by panda's .plot
                            
    Returns:
        Axis: matplotlib axis
        """
    
    # sorting with asc=false for correct order of bars
    
    if intercept==False:
        coeffs = coeffs.drop('intercept')
    
    if top_n is None:
        ## sort all features and set title
        plot_vals = coeffs.sort_values()
        title = "All Coefficients"

    else:
        ## rank the coeffs and select the top_n
        coeff_rank = coeffs.abs().rank().sort_values(ascending=False)
        top_n_features = coeff_rank.head(top_n)

        plot_vals = coeffs.loc[top_n_features.index].sort_values()
        ## sort features and keep top_n and set title
        title = f"Top {top_n} Largest Coefficients"
        
    ## plotting top N importances
    if threshold is not None:
        color_dict = get_colors_gt_lt(plot_vals, threshold=threshold,
                                      color_gt=color_gt,color_lt=color_lt,
                                      color_else=color_else)
        ## Getting color list and saving to plot_kws
        colors = plot_vals.index.map(color_dict)
        plot_kws.update({'color':colors})
        

    ax = plot_vals.plot(kind='barh', figsize=figsize,**plot_kws)
    ax.set(xlabel='Coefficient', 
           ylabel='Feature Names', 
           title=title)
    
    if threshold is not None:
        ln1 = ax.axvline(threshold,ls=':',color='black')

        from matplotlib.patches import Patch
        box_lt = Patch(color=color_lt)
        box_gt = Patch(color=color_gt)

        handles = [ln1,box_gt,box_lt]
        labels = [label_thresh,label_gt,label_lt]
        ax.legend(handles,labels, loc=legend_loc)
    ## return ax in case want to continue to update/modify figure
    return ax


# def annotate_bars(ax, ha='left',va='center',size=12,
#                 xytext=(4,0), textcoords='offset points'):
#     for bar in ax.patches:

#         ## calculate center of bar
#         bar_ax = bar.get_y() + bar.get_height()/2

#         ## get the value to annotate
#         val = bar.get_width()

#         # ha and va stand for the horizontal and vertical alignment
#         ax.annotate(f"{val:.3f}", (val,bar_ax),ha=ha,va=va,size=size,
#                     xytext=xytext, textcoords=textcoords)


## ADMIN VERSION 
def evaluate_classification_admin(model, X_train=None,y_train=None,X_test=None,y_test=None,
                            normalize='true',cmap='Blues', label= ': (Admin)', figsize=(10,5)):
    header="\tCLASSIFICATION REPORT " + label
    dashes='--'*40
    print(f"{dashes}\n{header}\n{dashes}")
    
    if (X_train is None) & (X_test is None):
        raise Exception("Must provide at least X_train & y_train or X_test and y_test")
    
    if (X_train is not None) & (y_train is not None):
        ## training data
        print(f"[i] Training Data:")
        y_pred_train = model.predict(X_train)
        report_train = metrics.classification_report(y_train, y_pred_train)
        print(report_train)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_train,y_train,
                                                      normalize=normalize, 
                                                      cmap=cmap,ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_train,y_train,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()

        plt.show()

    
        print(dashes)

        
    if (X_test is not None) & (y_test is not None):
        ## training data
        print(f"[i] Test Data:")
        y_pred_test = model.predict(X_test)
        report_test = metrics.classification_report(y_test, y_pred_test)
        print(report_test)

        fig,ax = plt.subplots(figsize=figsize,ncols=2)
        metrics.ConfusionMatrixDisplay.from_estimator(model,X_test,y_test,
                                                      normalize=normalize, 
                                                      cmap=cmap, ax=ax[0])
        try:
            metrics.RocCurveDisplay.from_estimator(model,X_test,y_test,ax=ax[1])
            ax[1].plot([0,1],[0,1],ls=':')
            ax[1].grid()
        except:
            fig.delaxes(ax[1])
        fig.tight_layout()
        plt.show()
        
        
        
def find_outliers_Z(data, verbose=True):
    import pandas as pd
    import numpy as np
    import scipy.stats as stats
    outliers = np.abs(stats.zscore(data))>3
    
    
    if verbose:
        n = len(outliers)
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum()/n*100:.2f}%) using Z-scores.")

    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers

def find_outliers_IQR(data, verbose=True):
    import pandas as pd
    import numpy as np
    q3 = np.quantile(data,.75)
    q1 = np.quantile(data,.25)

    IQR = q3 - q1
    upper_threshold = q3 + 1.5*IQR
    lower_threshold = q1 - 1.5*IQR
    
    outliers = (data<lower_threshold) | (data>upper_threshold)
    if verbose:
        n = len(outliers)
    
    
        print(f"- {outliers.sum():,} outliers found in {data.name} out of {n:,} rows ({outliers.sum()/n*100:.2f}%) using IQR.")
        
    outliers = pd.Series(outliers, index=data.index, name=data.name)
    return outliers




def remove_outliers(df_,method='iqr', subset=None, verbose=2):
    """Returns a copy of the input df with outleirs removed from all
    columns using the selected method (either 'iqr' or 'z'/'zscore')
    
    Arguments:
        df_ (Frame): Dataframe to copy and remove outleirs from
        method (str): Method of outlier removal. Options are 'iqr' or 'z' (default is 'iqr')
        subset (list or None): List of column names to remove outliers from. If None, uses all numeric columns.
        verbose (bool, int): If verbose==1, print only overall summary. If verbose==2, print detailed summary"""
    import pandas as pd
    ## Make a cope of input dataframe  
    df = df_.copy()
    
    ## Set verbose_func for calls to outleir funcs
    if verbose==2:
        verbose_func = True
    else:
        verbose_func=False
        
    ## Set outlier removal function and name
    if method.lower()=='iqr':
        find_outlier_func = find_outliers_IQR
        method_name = "IQR rule"
    elif 'z' in method.lower():
        find_outlier_func = find_outliers_Z
        method_name = 'Z_score rule'
    else:
        raise Exception('[!] Method must be either "iqr" or "z".')
        
    ## Set list of cols to remove outliers from
    if subset is None:
        col_list = df.select_dtypes('number').columns
    elif isinstance(subset,str):
        col_list = [subset]
    elif isinstance(subset, list):
        col_list = subset
    else:
        raise Exception("[!] subset must be None, a single string, or a list of strings.")

    

    
    ## Empty dict for both types of outliers
    outliers = {}

    ## Use both functions to see the comparison for # of outliers
    for col in col_list:
        idx_outliers = find_outlier_func(df[col],verbose=verbose_func)
        outliers[col] = idx_outliers

    
    ## Getting final df of all outliers to get 1 final T/F index
    outliers_combined = pd.DataFrame(outliers).any(axis=1)
    
    if verbose:
        n = len(outliers_combined)
        print(f"\n[i] Overall, {outliers_combined.sum():,} rows out of {n:,}({outliers_combined.sum()/n*100:.2f}%) were removed as outliers using {method_name}.")
    
    
    # remove_outliers 
    df_clean = df[~outliers_combined].copy()
    return df_clean
      
    
    
def evaluate_ols(result,X_train_df, y_train, show_summary=True):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    try:
        display(result.summary())
    except:
        pass
    
    ## save residuals from result
    y_pred = result.predict(X_train_df)
    resid = y_train - y_pred
    
    fig, axes = plt.subplots(ncols=2,figsize=(12,5))
    
    ## Normality 
    sm.graphics.qqplot(resid,line='45',fit=True,ax=axes[0]);
    
    ## Homoscedasticity
    ax = axes[1]
    ax.scatter(y_pred, resid, edgecolor='white',lw=1)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value');
    plt.tight_layout()
    
    
def plot_coeffs(result, drop_params=[], include_const=False, figsize=(6,10),
                const_name='const', title='Coefficients'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import warnings
    
    if include_const:
        drop_list = drop_params
    else:
        if const_name in result.params.index:
            drop_list = [*drop_params, const_name]
        else:
            drop_list = drop_params
    
    ### Make sure all drop_list in params
    dne_cols = list(filter(lambda x: x not in result.params,drop_list))
    drop_list = list(filter(lambda x: x in result.params, drop_list))
    if len(dne_cols)>0:
        warnings.warn(f"[!] These features were not found in the params: {dne_cols}")
    parms_to_plot = result.params.drop(drop_list).sort_values()
    
    if len(parms_to_plot)>1: 
        fig, ax = plt.subplots(ncols=1,figsize=figsize)
        parms_to_plot.plot(kind='barh', ax=ax)
        ax.axvline(color='k')
        ax.set(title=title, xlabel='Coefficient',ylabel='Feature')
        return fig
    else:
        import warnings
        warnings.warn("\n[!] There were no coefficients to plot.")
        
                     
        
def get_coeffs(reg,X_train,intercept_name='const',name=None):
    """Extracts the coefficients from a scikit-learn LinearRegression or LogisticRegression"""
    import pandas as pd
    coeffs = pd.Series(reg.coef_.flatten(),index=X_train.columns,name=name)
    try:
        intercept = reg.intercept_
        if intercept!=0:
            coeffs.loc[intercept_name] = intercept
    except:
        pass

    return coeffs


def get_importance(tree, X_train_df, top_n=20,figsize=(10,10),plot=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_importance = pd.Series(tree.feature_importances_,
                              index=X_train_df.columns)

    if plot:
        plt.figure(figsize=figsize)
        df_importance.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',title='Feature Importances',
    ylabel='Feature',)  
    else: 
        df_importance.sort_values(ascending=False)
    return df_importance
    
    
    
def evaluate_regression_resids(model,X_test_df, y_test, X_train_df=None, y_train=None,
                       single_split_label='Test',return_scores=False):
    """Plots a Q-Q Plot and residual plot for a statsmodels OLS regression.
    """
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn import metrics
    import pandas as pd
    
    # start list of results (will become dataframe)
    results = [['Metric','Split','Score']]
    
    ## get metrics for training data if provided
    if (X_train_df is not None) & (y_train is not None):
        other_split_label="Test"
        ## save residuals from result
        y_hat_train = model.predict(X_train_df)
        resid_train = y_train - y_hat_train
        rmse_train = metrics.mean_squared_error(y_train,y_hat_train,
                                                squared=False)

        r2_train = metrics.r2_score(y_train,y_hat_train)
        results.append(['R^2','Train',r2_train])
        results.append(['RMSE','Train',rmse_train])

    else:
        other_split_label = single_split_label


    ## save residuals from result
    y_hat_test = model.predict(X_test_df)
    resid_test = y_test - y_hat_test
    rmse_test = metrics.mean_squared_error(y_test,y_hat_test,
                                            squared=False)
    r2_test = metrics.r2_score(y_test,y_hat_test)
    results.append(['RMSE',other_split_label,rmse_test])
    results.append(['R^2',other_split_label,r2_test])
    
    
    ## Prepare final dataframe of results
    results_df = pd.DataFrame(results[1:], columns=results[0])
    results_df = results_df.sort_values('Metric',ascending=False)


    

    ## PLOT RESIDUAL DIAGNOSTICS
    fig, axes = plt.subplots(ncols=2,figsize=(12,5))
    
    ## Normality 
    sm.graphics.qqplot(resid_test,line='45',fit=True,ax=axes[0]);
    
    ## Homoscedasticity
    ax = axes[1]
    ax.scatter(y_hat_test, resid_test, edgecolor='white',lw=1)
    ax.axhline(0,zorder=0)
    ax.set(ylabel='Residuals',xlabel='Predicted Value',);
    fig.suptitle(f"Regression Diagnostics for {other_split_label} Data")
    plt.tight_layout()
    plt.show()
    
    display(results_df.set_index(['Metric','Split']).round(2))
    
    if return_scores:
        return results_df
        
        
        
### BELOW FROM prisoner_project_functions.py
def evaluate_regression(model,X_test,y_test,  X_train=None,y_train=None, get_params=True,
                       sort_params=True,ascending=True, warn=False):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn import metrics
    import pandas as pd
    if warn:
        import warnings
        warnings.warn('[!] This function has replaced the previous one. For the vers that diagnoses redisduals, run evaluate_regression_resids')
    
    #### 
    results = []
    y_hat_train = model.predict(X_train)
    r2_train = r2_score(y_train,y_hat_train)
    rmse_train = mean_squared_error(y_train,y_hat_train, squared=False)
    results.append({'Data':'Train', 'R^2':r2_train, "RMSE": rmse_train})
    
    y_hat_test = model.predict(X_test)
    r2_test = r2_score(y_test,y_hat_test)
    rmse_test = mean_squared_error(y_test,y_hat_test, squared=False)
    results.append({'Data':'Test', 'R^2':r2_test, "RMSE": rmse_test})
    
    results_df = pd.DataFrame(results).round(3).set_index('Data')
    results_df.index.name=None
    results_df.loc['Delta'] = results_df.loc['Test'] - results_df.loc['Train']
    results_df = results_df.T
    
    print(results_df)
                    
    if get_params:
        ## if a regression with coef_
        if hasattr(model, 'coef_'):
            params = pd.Series(model.coef_, index= X_train.columns,
                              name='Coefficients')
            params.loc['intercept'] = model.intercept_

        ## if a tree model with feature importance
        elif hasattr(model, 'feature_importances_'):
            params = pd.Series(model.feature_importances_,
                              index=X_train.columns, name='Feature Importance')
            
        else:
            print('[!] Could not extract coefficients or feature importances from model.')
    if sort_params:
        return params.sort_values(ascending=ascending)
    else:
        return params


    
    

def evaluate_classification(model, X_test,y_test,cmap='Greens',
                            normalize='true',classes=None,figsize=(10,4),
                            X_train = None, y_train = None,label='Test Data',
                            return_report=False):
    """Evaluates a scikit-learn binary classification model.

    Args:
        model (classifier): any sklearn classification model.
        X_test_tf (Frame or Array): X data
        y_test (Series or Array): y data
        cmap (str, optional): Colormap for confusion matrix. Defaults to 'Greens'.
        normalize (str, optional): normalize argument for plot_confusion_matrix. 
                                    Defaults to 'true'.
        classes (list, optional): List of class names for display. Defaults to None.
        figsize (tuple, optional): figure size Defaults to (8,4).
        
        X_train (Frame or Array, optional): If provided, compare model.score 
                                for train and test. Defaults to None.
        y_train (Series or Array, optional): If provided, compare model.score 
                                for train and test. Defaults to None.
    """
    ## 
    from sklearn import metrics
    import numpy as np
    import matplotlib.pyplot as plt
#     if classes is None:
#         classes = np.unique(y_test)
    get_report(model,X_test,y_test,as_df=False,label=label,target_names=classes)
    
    ## Plot Confusion Matrid and roc curve
    fig,ax = plt.subplots(ncols=2, figsize=figsize)
    metrics.ConfusionMatrixDisplay.from_estimator(model, X_test,y_test,cmap=cmap, 
                                  normalize=normalize,display_labels=classes,
                                 ax=ax[0])
    
    ## if roc curve erorrs, delete second ax
    try:
        curve = metrics.plot_roc_curve(model,X_test,y_test,ax=ax[1])
        curve.ax_.grid()
        curve.ax_.plot([0,1],[0,1],ls=':')
        fig.tight_layout()
    except:
        fig.delaxes(ax[1])
        
    plt.show()
    
    ## Add comparing Scores if X_train and y_train provided.
    if (X_train is not None) & (y_train is not None):
        print(f"Training Score = {model.score(X_train,y_train):.2f}")
        print(f"Test Score = {model.score(X_test,y_test):.2f}")
        
    if return_report:
        return get_report(model,X_test,y_test,as_df=True,label=label)
        
        

def get_time(verbose=False):
    import datetime as dt
    import time
    """Helper function to return current time.
    Uses tzlocal to display time in local tz, if available."""
    import tzlocal as tz
    try: 
        now =  dt.datetime.now(tz.get_localzone())
        tic = time.time()
    except:
        now = dt.datetime.now()
        tic = time.time()
        print("[!] Returning time without tzlocal.")       
    return now,tic
        
    
def get_report(model,X_test,y_test,as_df=False,target_names=None,label="TEST DATA"):
    """Get classification report from sklearn and converts to DataFrame"""
    ## Get Preds and report
    import pandas as pd
    from sklearn import metrics
    y_hat_test = model.predict(X_test)
    scores = metrics.classification_report(y_test, y_hat_test,
                                          output_dict=as_df,
                                          target_names=target_names) 
    ## convert to df if as_df
    if as_df:
        report = pd.DataFrame(scores).T.round(2)
        report.iloc[2,[0,1,3]] = ''
        return report
    else:
        header="\tCLASSIFICATION REPORT"
        if len(label)>0:
            header += f" - {label}"
        dashes='---'*20
        print(f"{dashes}\n{header}\n{dashes}")
        print(scores)
        
        
        
    
def fit_and_time_model(model, X_train,y_train,X_test,y_test,
                      fit_kws={}, scoring="accuracy",normalize='true',
                       fmt="%m/%d/%y-%T", verbose=True):
    """[Fits the provided model and evaluates using provided data.

    Args:
        model (classifier]): Initialized Model to fit and evaluate
        X_train (df/matrix): [description]
        y_train (series/array): [description]
        X_test (df/matrix): [description]
        y_test (series/array): [description]
        fit_kws (dict, optional): Kwargs for .fit. Defaults to {}.
        scoring (str, optional): Scoring metric to use. Defaults to "accuracy".
        normalize (str, optional): Normalize confusion matrix. Defaults to 'true'.
        fmt (str, optional): Time format. Defaults to "%m/%d/%y-%T".
        verbose (bool, optional): [description]. Defaults to True.

    Raises:
        Exception: [description]
    """
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import pandas as pd
    if X_test.ndim==1:
        raise Exception('The arg order has changed to X_train,y_train,X_test,y_test')

    ## Time
    start,tic = get_time()
    if verbose: 
        print(f"[i] Training started at {start.strftime(fmt)}:")
        
    model.fit(X_train, y_train,**fit_kws)
    
    ## Calc stop time and elapse
    stop,toc = get_time()
    elapsed = toc-tic


            
            
    ## Get model scores
    scorer = metrics.get_scorer(scoring)
    scores_dict ={f'Train':scorer(model,X_train,y_train),  
                  f'Test':scorer(model, X_test,y_test)}
    scores_dict['Difference'] = scores_dict['Train'] - scores_dict['Test']
    scores_df = pd.DataFrame(scores_dict,index=[scoring])
    
    ## Time and report back
    if verbose:
#         print(f"[i] Training completed at {stop.strftime(fmt)}")
        if elapsed >120:
            print(f"\tTraining time was {elapsed/60:.4f} minutes.")
        else:
            print(f"\tTraining time was {elapsed:.4f} seconds.")
    print("\n",scores_df.round(2),"\n")
    
    ## Plot Confusion Matrix and display classification report
    get_report(model,X_test,y_test,as_df=False)
    
    fig,ax = plt.subplots(figsize=(10,5),ncols=2)
    metrics.plot_confusion_matrix(model,X_test,y_test,normalize=normalize,
                                  cmap='Blues',ax=ax[0])

    try:
        metrics.plot_roc_curve(model,X_test,y_test,ax=ax[1])
        ax[1].plot([0,1],[0,1],ls=':')
        ax[1].grid()
    except: 
        fig.delaxes(ax[1])
    fig.tight_layout()
    plt.show()
    return model


# def evaluate_classification(model, X_test,y_test,normalize='true'):
#     """Plot Confusion Matrix and display classification report"""
#     get_report(model,X_test,y_test,as_df=False)
    
#     fig,ax = plt.subplots(figsize=(10,5),ncols=2)
#     metrics.plot_confusion_matrix(model,X_test,y_test,normalize=normalize,
#                                   cmap='Blues',ax=ax[0])
#     metrics.plot_roc_curve(model,X_test,y_test,ax=ax[1])
#     ax[1].plot([0,1],[0,1],ls=':')
#     ax[1].grid()
#     fig.tight_layout()
#     plt.show()




def evaluate_grid(grid,X_test,y_test,X_train=None,y_train=None):
    print('The best parameters were:')
    print("\t",grid.best_params_)
    
    model = grid.best_estimator_    

    print('\n[i] Classification Report')
    evaluate_classification(model, X_test,y_test,X_train=X_train,y_train=y_train)
    
    
    
def get_importance(tree, X_train_df, top_n=20,figsize=(10,10),plot=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    df_importance = pd.Series(tree.feature_importances_,
                              index=X_train_df.columns)

    if plot:
        plt.figure(figsize=figsize)
        df_importance.sort_values(ascending=True).tail(top_n).plot(
        kind='barh',title='Feature Importances',
    ylabel='Feature',)  
    else: 
        df_importance.sort_values(ascending=False)
    return df_importance



def show_tree(clf,X_train_df,figsize=(60,25),class_names=['Died','Survived'],
              savefig=False,fname='titanic_tree.pdf',max_depth=None,):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.tree import plot_tree
    fig,ax = plt.subplots(figsize=figsize)
    plot_tree(clf,filled=True,rounded=True,proportion=True,
              feature_names=X_train_df.columns,
              class_names=class_names,ax=ax);
    fig.tight_layout()
    
    if savefig:
        fig.savefig(fname, dpi=300,orientation='landscape')
    return fig



def compare_importances(*importances,sort_index=True,sort_col=0,show_bar=False):
    """Accepts Series of feature importances to concat.
    
    Args:
        *importances (Seires): seires to concat (recommended to pre-set names of Series)
        sort_index (bool, default=True): return series sorted by index. 
                            If False, sort seires by sort_col  #
        sort_col (int, default=0): If sort_index=False, sort df by this column #
        show_bar (bool, default=False): If show_bar, returns a pandas styler instead of df
                                        with the importances plotted as bar graphs
        
    Returns:
        DataFrame: featutre importances     
    
        """
    import matplotlib.pyplot as plt
    import pandas as pd
    ## Concat Importances
    compare_importances = pd.concat(importances,axis=1)
    
    ## Sort DF by index or by sort_col
    if sort_index:
        sort_col_name = 'Index'
        compare_importances = compare_importances.sort_index()
    else:
        sort_col_name = compare_importances.columns[sort_col]
        compare_importances= compare_importances.sort_values(sort_col_name,ascending=False)
        
    ## If show bar, return pandas styler with in-cell bargraphs
    if show_bar:
        return compare_importances.style.bar().set_caption(f'Feature Importances - sorted by {sort_col_name}')
    else:
        return compare_importances


## update function to return 
def get_logreg_coefficients(model,X_train,units = "log-odds"):
    """Returns model coefficients. 
    
    Args:
        model: sklearn model with the .coef_ attribute. 
        X_train: dataframe with the feature names as the .columns
        units (str): Can be ['log-odds','odds','prob']
        """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    options = ['log-odds','odds','prob']
    
    if units not in options:
        raise Exception(f'units must be one of {options}')
        
    coeffs = pd.Series(model.coef_.flatten(), index=X_train.columns)
    coeffs['intercept'] = model.intercept_[0]
    
    if units=='odds':
        coeffs = np.exp(coeffs)
        
    elif units=='prob':
        coeffs = np.exp(coeffs)
        coeffs = coeffs/(1+coeffs)
        

    coeffs.name=units
    return coeffs


def evaluate_linreg(model, X_train,y_train, X_test,y_test, get_coeffs=True):
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    results = []
    y_hat_train = model.predict(X_train)
    r2_train = r2_score(y_train,y_hat_train)
    rmse_train = mean_squared_error(y_train,y_hat_train, squared=False)
    results.append({'Data':'Train', 'R^2':r2_train, "RMSE": rmse_train})
    
    y_hat_test = model.predict(X_test)
    r2_test = r2_score(y_test,y_hat_test)
    rmse_test = mean_squared_error(y_test,y_hat_test, squared=False)
    results.append({'Data':'Test', 'R^2':r2_test, "RMSE": rmse_test})
    
    results_df = pd.DataFrame(results).round(3).set_index('Data')
    results_df.index.name=None
    print(results_df)
                    
    if get_coeffs:
        coeffs = pd.Series(model.coef_, index= X_train.columns)
        coeffs.loc['intercept'] = model.intercept_
        return coeffs

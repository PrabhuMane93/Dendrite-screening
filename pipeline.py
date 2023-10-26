from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def grid_search_cv_pipeline(pipeline_obj,data):
    for model in pipeline_obj:
        model_param = data['design_state_data']['algorithms'][model]
        if model == 'RandomForestRegressor':
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {'n_estimators'     : [int(model_param["min_trees"])                      ,  int(model_param["max_trees"])],
                                      'max_depth'        : [int(model_param["min_depth"])                      ,  int(model_param["max_depth"])],
                                      'min_samples_leaf' : [int(model_param["min_samples_per_leaf_min_value"]) ,  int(model_param["min_samples_per_leaf_max_value"])]
                                     }
            ))
            pipeline_obj[model] = reg

        elif model == "LinearRegression":
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {}
            ))
            pipeline_obj[model] = reg        
    
        elif model == "RidgeRegression":
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {'max_iter' : [int(model_param["min_iter"])       ,   int(model_param["max_iter"])],
                                      'alpha'    : [float(model_param["min_regparam"]) ,   float(model_param["max_regparam"])]}
            ))
            pipeline_obj[model] = reg 
    
        elif model == "LassoRegression":
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {'max_iter' : [int(model_param["min_iter"])       ,   int(model_param["max_iter"])],
                                      'alpha'    : [float(model_param["min_regparam"]) ,   float(model_param["max_regparam"])]}
            ))
            pipeline_obj[model] = reg 
    
        elif model == "ElasticNetRegression":
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {'max_iter' : [int(model_param["min_iter"])        ,   int(model_param["max_iter"])],
                                      'alpha'    : [float(model_param["min_regparam"])  ,   float(model_param["max_regparam"])],
                                      'l1_ratio' : [float(model_param["min_elasticnet"]),   float(model_param["min_elasticnet"])]}
            ))
            pipeline_obj[model] = reg 
    
        elif model == "DecisionTreeRegressor":
            reg = make_pipeline(StandardScaler(),GridSearchCV(
                        estimator = pipeline_obj[model],
                        param_grid = {'max_depth'        : [int(model_param["min_depth"]) , int(model_param["max_depth"])],
                                      'min_samples_leaf' : model_param['min_samples_per_leaf']}
            ))
            pipeline_obj[model] = reg 
    return pipeline_obj
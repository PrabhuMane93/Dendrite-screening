from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor

def build_models(models, data):

    model_pipelines = {}
    for model in models:
        model_param = data['design_state_data']['algorithms'][model]
        
        if model == "RandomForestRegressor":
            model = RandomForestRegressor()
            model_pipelines["RandomForestRegressor"] = model
    
    
        elif model == "LinearRegression":
            model = LinearRegression()
            model_pipelines["LinearRegression"] = model
        
    
        elif model == "RidgeRegression":
            model = Ridge(
                    alpha = float(model_param["regularization_term"]))
            model_pipelines["RidgeRegression"] = model
    
        elif model == "LassoRegression":
            model = Lasso(
                    max_iter = int(model_param["max_iter"]),
                    alpha = float(model_param["regularization_term"]))
            model_pipelines["LassoRegression"] = model
    
        elif model == "ElasticNetRegression":
            model = ElasticNet(
                    max_iter = int(model_param["max_iter"]),
                    alpha = float(model_param["regularization_term"]))
            model_pipelines["ElasticNetRegression"] = model
    
        elif model == "DecisionTreeRegressor":
            model = DecisionTreeRegressor(
                    max_depth = int(model_param["max_depth"]))
            model_pipelines["DecisionTreeRegressor"] = model

    return model_pipelines

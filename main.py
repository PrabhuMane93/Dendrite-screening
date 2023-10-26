import json
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from transform import transform_df
from feature_reduction import reduce
from models import build_models
from pipeline import grid_search_cv_pipeline

class screening_test:
    def __init__(self, path_to_json, path_to_csv):

        with open(path_to_json, 'r') as file:
            self.data = json.load(file)

        df = pd.read_csv(path_to_csv)
        target = self.data['design_state_data']['target']['target']
        prediction_type = self.data['design_state_data']['target']['prediction_type']
        features = self.data['design_state_data']["feature_handling"]
        df = transform_df(df, features)

        df = reduce(target, df, self.data['design_state_data']['feature_reduction'])

        self.models = ["RandomForestRegressor", "GBTRegressor", "LinearRegression", "RidgeRegression", "LassoRegression", "ElasticNetRegression", "DecisionTreeRegressor"]

        self.X = df.drop(columns = [target]).values
        self.y = df[target].values

    def run(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,random_state=0)

        model_pipelines = build_models(self.models, self.data)
        pipelines = grid_search_cv_pipeline(model_pipelines, self.data)
        print("\n\n")
        print("-----------------------------------------------------------------------------------------------------------------------------")
        print("                                                         Model Metrics")
        print("-----------------------------------------------------------------------------------------------------------------------------")
        print("\n")
        for i in pipelines:
            pipelines[i].fit(X_train, y_train)
            print(f"Accuracy of {i} : {pipelines[i].score(X_test, y_test)}")
            best_estimator = pipelines[i].named_steps['gridsearchcv'].best_estimator_
            print(f"Model metrics are : {best_estimator}")
            print("\n")

if __name__ == "__main__":
    path_to_json = sys.argv[1]
    path_to_csv  = sys.argv[2]

    screening_test_obj = screening_test(path_to_json, path_to_csv)
    screening_test_obj.run()
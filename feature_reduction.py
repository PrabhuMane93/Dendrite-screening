import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 

def reduce(target, df, reduction_dictionary):

    if reduction_dictionary["feature_reduction_method"] == "Correlation with target":
        num_of_features_to_keep = reduction_dictionary["num_of_features_to_keep"]
        correlations = df.corr()[target]
        sorted_features = correlations.abs().sort_values(ascending=False)
        top_features = sorted_features.head(int(num_of_features_to_keep)+1).index
        new_df = df[top_features]
        new_df[target] = df[target]
        return new_df

    elif reduction_dictionary["feature_reduction_method"] == "Tree-based":
        num_of_features_to_keep = reduction_dictionary["num_of_features_to_keep"]
        depth_of_trees = reduction_dictionary["depth_of_trees"]
        num_of_trees = reduction_dictionary["num_of_trees"]
        model = RandomForestRegressor(n_estimators = int(num_of_trees), max_depth = int(depth_of_trees))
        model.fit(df.drop(columns = [target]).values, df[target].values)
        feature_importance = model.feature_importances_
        sorted_indices = feature_importance.argsort()[::-1]
        sorted_features = df.drop(columns = [target]).columns[sorted_indices]
        selected_features = sorted_features[:int(num_of_features_to_keep)]
        new_df = df[selected_features]
        new_df[target] = df[target]
        return new_df
    
    elif reduction_dictionary["feature_reduction_method"] == "Principal Component Analysis":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.drop(columns = [target]).values)
        pca = PCA(n_components=int(reduction_dictionary["num_of_features_to_keep"]))
        pca_result = pca.fit_transform(scaled_data)
        components = pca.components_
        new_column_names = [f'PC{i+1}' for i in range(int(reduction_dictionary["num_of_features_to_keep"]))]
        pca_df = pd.DataFrame(data=pca_result, columns=new_column_names)
        pca_df[target] = df[target]
        return pca_df

    elif reduction_dictionary["feature_reduction_method"] == "No Reduction":
        return df
        
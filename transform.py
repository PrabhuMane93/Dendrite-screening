import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import HashingVectorizer
nltk.download('punkt')

def transform_df(df, features):
    for i in features:
        if features[i]["is_selected"]:
            if features[i]["feature_variable_type"]=="numerical":
                df[i] = pd.to_numeric(df[i])
                if features[i]["feature_details"]["impute_with"]=="Average of values":
                    mean  = df[i].mean()
                    df[i] = df[i].fillna(mean)
                else:
                    df[i] = df[i].fillna(int(features[i]["feature_details"]["impute_value"]))
           
            if features[i]["feature_variable_type"]=="text":
                df = tokenize_and_hash(df, i)
        else:
            df = df.drop(columns=[i])   
    return df

def tokenize_and_hash(df, column_name):
    #tokenize the text column
    tokens = df[column_name].apply(word_tokenize)
    #hash the tokens
    hash_vectorizer = HashingVectorizer(n_features=5)       #Default number of columns for tokenization is 5
    hashed_features = hash_vectorizer.transform(tokens.apply(' '.join))
    hashed_df = pd.DataFrame(hashed_features.toarray(), columns=[f'{column_name}_feature_{i+1}' for i in range(5)])
    #replace the text columns with newly created hashs
    df = df.drop(columns = [column_name])
    return pd.concat([df, hashed_df], axis=1)
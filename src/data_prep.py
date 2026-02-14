from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocesar(X, y):

    mapping = {'No': 0, 'Yes': 1}
    y = y.map(mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, drop='if_binary'), cat_cols)
    ])
    
    preprocessor.set_output(transform="pandas")
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)


    return X_train_proc, X_test_proc, y_train, y_test, pipeline
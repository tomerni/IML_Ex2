import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X: np.array, y: np.array) -> tuple:
    """
    fits the linear regression according to the normal equations solution
    we saw in class
    :param X: the design matrix - m rows, d cols
    :param y: the response vectors - m rows
    :return: w_hat - the coefficients vector (d rows), sigma_val - the singular
    values of X
    """
    U, sigma_val, V = np.linalg.svd(X)
    # sigma_matrix = np.zeros((U.shape[0], V.shape[1]), np.float)
    # np.fill_diagonal(sigma_matrix, sigma_val)
    X_pseudo_inverse = np.linalg.pinv(X)
    w_hat = X_pseudo_inverse.dot(y)
    return w_hat, sigma_val


def predict(X: np.array, w: np.array) -> np.array:
    """
    predicts the results according to the coefficients vector.
    :param X: the design matrix - m rows, d, cols
    :param w: the coefficients vector - d row
    :return: the predicted response vector - m rows
    """
    return np.dot(X, w)


def mse(response_vector: np.array, prediction_vector: np.array) -> float:
    """
    calculates the MSE
    :param response_vector: the true responses
    :param prediction_vector: the predicted responses
    :return: the value of the MSE
    """
    return ((response_vector - prediction_vector) ** 2).mean(axis=0)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    loads the data from the csv to a DataFrame and preprocesses it.
    The categorical features are - zipcode, date (maybe), long/lat
    :param csv_path:
    :return:
    """
    df = pd.read_csv(csv_path)
    df = preprocessing(df)
    return df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    # TODO: add the intercept col, maybe remove zipcode, deal with illegal
    #       values, maybe remove date?
    df.insert(0, 'intercept', 1)
    df = df.drop('id', axis=1)
    df = df.drop('zipcode', axis=1)
    df = df.drop('date', axis=1)
    # df = df.drop('yr_renovated', axis=1)
    # df = df.drop('yr_built', axis=1)
    # df = df.drop('lat', axis=1)
    # df = df.drop('long', axis=1)
    df = df[df['price'] > 0]
    df = df[df['price'] != 'nan']
    return df


def plot_singular_values():
    """
    plots the features and their singular values
    :return:
    """
    # TODO: figure out what this function gets
    pass


def getting_ready() -> pd.DataFrame:
    """
    loads the csv and pre-processes it
    :return: the processed data frame
    """
    PATH = r"./kc_house_data.csv"
    df = load_data(PATH)
    # TODO: make this functional
    # plot_singular_values()
    return df


def train(df: pd.DataFrame):
    """
    trains the model according to question 16 and shows the predictions and MSE
    :param df: the design matrix
    """
    test_df = df.sample(frac=0.25)
    response_vector = test_df['price'].to_numpy().astype(np.float)
    test_df = test_df.drop('price', axis=1)
    training_df = df.drop(test_df.index)
    test_arr = test_df.to_numpy()
    mse_arr = list()
    for i in range(1, 101):
        training_arr = training_df.head(int(len(training_df) * (i / 100)))
        design_matrix = training_arr.drop('price', axis=1).to_numpy().astype(np.float)
        cur_response_vector = training_arr['price'].to_numpy().astype(np.float)
        coefficients_vector = fit_linear_regression(design_matrix,
                                                    cur_response_vector)[0]
        predict_vector = predict(test_arr, coefficients_vector)
        x = mse(response_vector, predict_vector)
        mse_arr.append(x)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('percent')
    ax.set_ylabel('MSE')
    ax.set_title("MSE as a function of %")
    ax.plot(np.arange(1, 101), np.array(mse_arr))
    fig.show()
    fig.savefig(r'./images/mse.png')


def feature_evaluation(design_matrix: pd.DataFrame, response_vector: np.array):
    """
    evaluates the numeric features according to question 17 and shows the
    correlation between them and the price
    :param design_matrix
    :param response_vector
    """
    numeric_features = ["sqft_living", "sqft_lot", "sqft_above",
                        "sqft_basement", "sqft_living15", "sqft_lot15",
                        "bedrooms", "bathrooms", "floors", "condition",
                        "grade"]
    response_vector_deviation = np.std(response_vector)
    for feature in numeric_features:
        feature_vector = design_matrix[feature].to_numpy().astype(np.float)
        divisor = response_vector_deviation * np.std(feature_vector)
        cov = np.cov(response_vector, feature_vector)[0, 1]
        pearson_correlation = cov/divisor
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xlabel('{}'.format(feature))
        ax.set_ylabel('price')
        ax.set_title("correlation between {} and price. Pearson Correlation = "
                     "{}".format(feature, pearson_correlation))
        ax.scatter(feature_vector, response_vector)
        fig.show()
        fig.savefig(r'./images/{}_correlation.png'.format(feature))


if __name__ == '__main__':
    design_matrix = getting_ready()
    train(design_matrix)
    # response_vector = design_matrix['price'].to_numpy().astype(np.float)
    # mat = design_matrix.drop('price', axis=1)
    # feature_evaluation(mat, response_vector)
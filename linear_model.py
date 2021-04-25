import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUMERIC_FEATURES = ["sqft_living", "sqft_lot", "sqft_above",
                    "sqft_basement", "sqft_living15", "sqft_lot15",
                    "bedrooms", "bathrooms", "floors", "condition",
                    "grade", "waterfront", "view", "yr_renovated"]

COLS_TO_REMOVE = ['id', 'date', 'long', 'zipcode', 'yr_built', "sqft_lot",
                  "sqft_above"]


def fit_linear_regression(X: np.array, y: np.array) -> tuple:
    """
    fits the linear regression according to the normal equations solution
    we saw in class
    :param X: the design matrix - m rows, d cols
    :param y: the response vectors - m rows
    :return: w_hat - the coefficients vector (d rows), sigma_val - the singular
    values of X
    """
    sigma_val = np.linalg.svd(X, compute_uv=False)
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
    The categorical features are - zipcode, date, long, lat, yr_built,
    yr_renovated.
    I decided to drop the zipcode, date, long and yr_built columns.
    zipcode - after reading online and observing the data I found out that
    there is a very weak correlation between the zipcode and the price.
    date - the date is not correlated linearly with the price. Moreover, in the
    same date a very expensive house and a very cheap house could be sold, so
    I thought that it will add a lot of randomness and noise to the model.
    lat and long - in a big city (and even on small one) there are vast
    differences between houses in the same long and lat. This is because those
    fields aren't accurate enough. I think that they will add a lot of noise
    to the model. After I observed the correlation between those field and the
    price, I found out that the long field has almost 0 correlation so I
    decided to drop it.
    Moreover, I dropped the sqft_above and sqft_basement because in the data
    you can see that sqft_above + sqft_basement = sqft_living.
    :param csv_path: the path of the csv
    :return: the processed data frame
    """
    df = pd.read_csv(csv_path)
    df = preprocessing(df)
    return df


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    preprocesses the data - drops irrelevant columns, inserts the intercept col
    filters out nan rows, filters out 0 bedrooms
    :param df: the filtered data frame
    :return:
    """
    df.insert(0, 'intercept', 1)  # adding the intercept
    df = df.drop(columns=COLS_TO_REMOVE, axis=1)  # drops unnecessary cols
    df = df[(df['price'] > 0) & (df['price'] != 'nan')]  # invalid data in
    # the target col
    df = df[df['bedrooms'] > 0]  # this condition removes all most all of the
    # invalid rows in the data
    df.loc[(df.yr_renovated != 0), 'yr_renovated'] = 1  # categorical field
    return df


def plot_singular_values(singular_values: np.array):
    """
    plots the features and their singular values
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel('number of singular value')
    ax.set_ylabel('singular value')
    ax.set_title("The singular values of the data")
    x_axis = np.arange(1, len(singular_values) + 1)
    ax.set_xticks(x_axis)
    ax.scatter(x_axis, singular_values)
    fig.show()
    fig.savefig(r'./images/singular_values.png')


def getting_ready() -> pd.DataFrame:
    """
    loads the csv and pre-processes it
    :return: the processed data frame
    """
    PATH = r"./kc_house_data.csv"
    df = load_data(PATH)
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
        design_matrix = training_arr.drop('price', axis=1).to_numpy().astype(
            np.float)
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

    response_vector_deviation = np.std(response_vector)
    for feature in NUMERIC_FEATURES:
        feature_vector = design_matrix[feature].to_numpy().astype(np.float)
        divisor = response_vector_deviation * np.std(feature_vector)
        cov = np.cov(response_vector, feature_vector)[0, 1]
        pearson_correlation = cov / divisor
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
    matrix = getting_ready()
    Sigma = np.linalg.svd(matrix.to_numpy().astype(np.float),
                          compute_uv=False)
    plot_singular_values(Sigma)
    response_vector = matrix['price'].to_numpy().astype(np.float)
    mat = matrix.drop('price', axis=1)
    feature_evaluation(mat, response_vector)

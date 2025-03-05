import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def train_weights(X, Y, alpha, model) -> np.ndarray:
    """ Do a five-fold cross validation to train weights for each subj """
    clone_model = sklearn.base.clone(model)
    n_subjs, n_imgs = X.shape
    stims = np.arange(n_imgs)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    Intercept = np.empty((k, n_subjs))
    Beta = np.empty((k, n_subjs))

    for fold, (train_index, _) in enumerate(kf.split(stims)):
        if hasattr(clone_model, 'alpha'):
            clone_model.alpha = alpha
        X_train = X[:, train_index].T
        Y_train = Y[train_index]
        clone_model.fit(X_train, Y_train)
        Intercept[fold] = clone_model.intercept_
        Beta[fold] = clone_model.coef_

    Intercept = np.mean(Intercept)
    Beta = np.mean(Beta, axis=0)
    return Intercept, Beta


def get_prediction(model, x, w, c, a) -> np.ndarray:
    """ Do evaluation of the predictivity using the fitted weights """
    model.coef_ = w
    model.intercept_ = c
    if hasattr(model, 'alpha'):
        model.alpha = a
    y_pred = model.predict(x.T)
    return y_pred


def fit_eval_alpha_model(model, X, Y, alpha) -> float:
    """ Do a five fold cross validation to fit the alpha """
    clone_model = sklearn.base.clone(model)
    n_subjs, n_imgs = X.shape
    stims = np.arange(n_imgs)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = np.empty(k)

    for fold, (train_index, test_index) in enumerate(kf.split(stims)):
        clone_model.alpha = alpha
        X_train = X[:, train_index].T; Y_train = Y[train_index]
        X_test = X[:, test_index].T; Y_test = Y[test_index]
        clone_model.fit(X_train, Y_train)
        y_pred = clone_model.predict(X_test)
        score = np.corrcoef(y_pred, Y_test)[0, 1]
        scores[fold] = score
    return np.nanmean(scores)


def find_best_alpha(X, Y, model) -> float:
    """ Get the optimal alpha for Regularized regression """
    clone_model = sklearn.base.clone(model)
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        score = fit_eval_alpha_model(clone_model, X, Y, alpha)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return best_alpha


def train_model_for_each(model, X_train: np.ndarray, Y_train: np.ndarray) -> tuple:
    """ Train the model for each met, subjs """

    clone_model = sklearn.base.clone(model)

    n_met, n_subjs, n_imgs = X_train.shape
    W = np.empty((n_met, n_subjs, n_subjs - 1))
    C = np.empty((n_met, n_subjs))
    A = np.empty((n_met, n_subjs))

    for met in range(n_met):
        for subj in range(n_subjs):
            other_subjs = [i for i in range(n_subjs) if i != subj]
            if isinstance(clone_model, LinearRegression):
                best_alpha = None
            else:
                best_alpha = find_best_alpha(
                    X = X_train[met, other_subjs, :],
                    Y = Y_train[met, subj, :],
                    model = clone_model,
                )

            C[met, subj], W[met, subj] = \
                train_weights(
                    X = X_train[met, other_subjs, :], 
                    Y = Y_train[met, subj, :],
                    alpha = best_alpha,
                    model = clone_model,
                )
            A[met, subj] = best_alpha

    return W, C, A


def test_model_for_each(model, X_test: np.ndarray, Y_test: np.ndarray,
                        W: np.ndarray, C: np.ndarray, A: np.ndarray,
                        ) -> np.ndarray:
    """ Test the model for each met, subjs """

    n_met, n_subjs, _ = Y_test.shape
    output = np.empty((n_met, n_subjs))

    for met in range(n_met):
        for subj in range(n_subjs):
            other_subjs = [i for i in range(n_subjs) if i != subj]
            y_pred = get_prediction(
                model=model,
                x = X_test[met, other_subjs, :],
                w = W[met, subj],
                c = C[met, subj],
                a = A[met, subj],
            )
            output[met, subj] = np.corrcoef(y_pred, Y_test[met, subj, :])[0, 1]

    return output

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from autogluon.core.models import AbstractModel
from pytorch_tabnet.tab_model import TabNetClassifier


class NeuralNetTabNet(AbstractModel):
    """
    TabNet custom model for AutoGluon Tabular (AutoGluon 1.5.0).
    Assumes all features are numeric/encoded.
    Handles NaNs via median imputation.
    """

    #def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None, **kwargs):
    def _fit(self, X, y, X_val=None, y_val=None, **kwargs):
        # Keep column order consistent between fit and predict
        self.feature_columns = list(X.columns)

        # TabNet does not accept NaNs -> impute
        self.imputer = SimpleImputer(strategy="median")
        X_train = self.imputer.fit_transform(X[self.feature_columns])
        y_train = y.astype(int).to_numpy()

        eval_set = None
        if X_val is not None and y_val is not None:
            Xv = self.imputer.transform(X_val[self.feature_columns])
            yv = y_val.astype(int).to_numpy()
            eval_set = [(Xv, yv)]

        # Model hyperparameters (you can adjust these)
        self.model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            seed=42,
            verbose=0
        )

        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=eval_set,
            eval_metric=["auc"],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

    #def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        #Xp = self.imputer.transform(X[self.feature_columns])
        #proba = self.model.predict_proba(Xp)  # shape (n, 2) for binary
        #return proba

    #def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
    #Xp = self.imputer.transform(X[self.feature_columns])
    #proba_2d = self.model.predict_proba(Xp)      # shape (n, 2)
    #return proba_2d[:, 1]                        # shape (n,)

    def _predict_proba(self, X, **kwargs):
        Xp = self.imputer.transform(X[self.feature_columns])
        proba_2d = self.model.predict_proba(Xp)   # (n, 2)
        return proba_2d[:, 1]                     # (n,)
                    # (n,)


    def _save(self, path: str) -> str:
        """
        Save model + preprocessing artifacts.
        """
        os.makedirs(path, exist_ok=True)

        # Save TabNet using its own method (robust)
        tabnet_path = os.path.join(path, "tabnet_model.zip")
        self.model.save_model(tabnet_path)

        # Save imputer + columns using numpy
        np.save(os.path.join(path, "feature_columns.npy"), np.array(self.feature_columns, dtype=object), allow_pickle=True)
        # Store imputer attributes (median stats) in a simple way
        np.save(os.path.join(path, "imputer_statistics_.npy"), self.imputer.statistics_, allow_pickle=True)

        # Let AutoGluon save the rest of AbstractModel metadata
        return super()._save(path)

    @classmethod
    def _load(cls, path: str):
        """
        Load model + preprocessing artifacts.
        """
        obj = super(NeuralNetTabNet, cls)._load(path)

        # Load feature columns + imputer stats
        obj.feature_columns = np.load(os.path.join(path, "feature_columns.npy"), allow_pickle=True).tolist()
        stats = np.load(os.path.join(path, "imputer_statistics_.npy"), allow_pickle=True)

        # Reconstruct imputer
        obj.imputer = SimpleImputer(strategy="median")
        obj.imputer.statistics_ = stats

        # Load TabNet
        obj.model = TabNetClassifier()
        obj.model.load_model(os.path.join(path, "tabnet_model.zip"))

        return obj

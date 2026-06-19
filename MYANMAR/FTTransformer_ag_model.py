# FTTransformer_ag_model.py
# AutoGluon custom model wrapper for rtdl-revisiting-models FTTransformer
# Improvements in this version:
# 1) Fix _predict_proba indentation (was over-indented)
# 2) Default to CPU-only for stability on Windows laptops (use_cuda_if_available=False)
# 3) Safer default batch_size=256 (reduce spikes)
# 4) Minor robustness: handle missing/unseen categories consistently; safer label mapping

import os
import json
import inspect
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from autogluon.core.models import AbstractModel

from rtdl_revisiting_models import FTTransformer as RTDLFTTransformer


# ---- JSON helper (MUST be top-level, not inside the class) ----
def _json_safe(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


class NeuralNetFTTransformer(AbstractModel):
    """
    FT-Transformer custom model for AutoGluon Tabular.
    Handles:
      - Numeric + categorical mixed tabular data
      - Median imputation for numeric
      - Simple category encoding with 0 reserved for missing/unseen
    """

    def _set_default_params(self):
        super()._set_default_params()

        # Training
        self._set_default_param_value("epochs", 30)
        self._set_default_param_value("batch_size", 256)  # safer than 512 on laptops
        self._set_default_param_value("lr", 1e-3)
        self._set_default_param_value("weight_decay", 1e-5)
        self._set_default_param_value("patience", 5)
        self._set_default_param_value("num_workers", 0)

        # Model size / backbone
        self._set_default_param_value("d_token", 192)
        self._set_default_param_value("n_blocks", 3)
        self._set_default_param_value("attention_n_heads", 8)
        self._set_default_param_value("attention_dropout", 0.2)
        self._set_default_param_value("ffn_dropout", 0.1)
        self._set_default_param_value("residual_dropout", 0.0)

        # Optional numeric standardization
        self._set_default_param_value("standardize_num", False)

        # Device
        # IMPORTANT: default CPU-only for stability (you can override True on Colab/Linux)
        self._set_default_param_value("use_cuda_if_available", False)

    def _infer_problem_type(self, y: pd.Series):
        uniq = pd.unique(y.dropna())
        n_classes = len(uniq)
        if n_classes == 2:
            return "binary", 2
        return "multiclass", n_classes

    def _get_cat_num_cols(self, X: pd.DataFrame):
        cat_cols = []
        for c in X.columns:
            dt = str(X[c].dtype)
            if dt in ("category", "object", "bool"):
                cat_cols.append(c)
        num_cols = [c for c in X.columns if c not in cat_cols]
        return cat_cols, num_cols

    def _build_fttransformer(self, n_num_features: int, cat_sizes: list[int], out_dim: int):
        p = self.params
        sig = inspect.signature(RTDLFTTransformer.__init__)
        params = sig.parameters

        cat_cardinalities = cat_sizes if len(cat_sizes) > 0 else []

        # NOTE: your environment appears to require ffn_d_hidden_multiplier
        backbone_kwargs = {
            "d_block": int(p["d_token"]),
            "n_blocks": int(p["n_blocks"]),
            "attention_n_heads": int(p["attention_n_heads"]),
            "attention_dropout": float(p["attention_dropout"]),
            "ffn_dropout": float(p["ffn_dropout"]),
            "residual_dropout": float(p["residual_dropout"]),
            "ffn_d_hidden_multiplier": 4,
        }

        # Signature A
        if "n_cont_features" in params and "cat_cardinalities" in params:
            return RTDLFTTransformer(
                n_cont_features=n_num_features,
                cat_cardinalities=cat_cardinalities,
                d_out=out_dim,
                **backbone_kwargs,
            )

        # Signature B (older)
        if "n_num_features" in params and "categories" in params:
            categories = None if len(cat_sizes) == 0 else cat_sizes
            return RTDLFTTransformer(
                n_num_features=n_num_features,
                categories=categories,
                d_token=int(p["d_token"]),
                n_blocks=int(p["n_blocks"]),
                attention_n_heads=int(p["attention_n_heads"]),
                attention_dropout=float(p["attention_dropout"]),
                ffn_dropout=float(p["ffn_dropout"]),
                residual_dropout=float(p["residual_dropout"]),
                d_out=out_dim,
            )

        raise RuntimeError(f"Unsupported FTTransformer signature: {sig}")

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        p = self.params

        use_cuda = bool(p.get("use_cuda_if_available", False))
        self.device_ = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

        self.feature_columns_ = list(X.columns)
        self.cat_cols_, self.num_cols_ = self._get_cat_num_cols(X)

        # ---------- Numeric ----------
        self.num_imputer_ = SimpleImputer(strategy="median")
        if len(self.num_cols_) > 0:
            X_num = self.num_imputer_.fit_transform(X[self.num_cols_].astype("float32"))
            if p.get("standardize_num", False):
                self.num_mean_ = X_num.mean(axis=0).astype("float32")
                self.num_std_ = (X_num.std(axis=0) + 1e-8).astype("float32")
                X_num = (X_num - self.num_mean_) / self.num_std_
            else:
                self.num_mean_, self.num_std_ = None, None
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)
            self.num_mean_, self.num_std_ = None, None

        # ---------- Categorical ----------
        self.cat_maps_ = {}
        self.cat_sizes_ = []
        if len(self.cat_cols_) > 0:
            X_cat_list = []
            for col in self.cat_cols_:
                s_obj = X[col].astype("object")
                non_missing = pd.Series(s_obj[~pd.isna(s_obj)]).astype(str)
                uniques = pd.unique(non_missing)

                # 0 reserved for missing/unseen
                mapping = {v: i + 1 for i, v in enumerate(uniques)}
                self.cat_maps_[col] = mapping
                self.cat_sizes_.append(len(mapping) + 1)

                encoded = np.zeros(len(s_obj), dtype=np.int64)
                mask = ~pd.isna(s_obj)
                if mask.any():
                    vals = pd.Series(s_obj[mask]).astype(str)
                    enc = vals.map(mapping).fillna(0).astype(np.int64).to_numpy()
                    encoded[mask.to_numpy()] = enc
                X_cat_list.append(encoded.reshape(-1, 1))

            X_cat = np.concatenate(X_cat_list, axis=1).astype(np.int64)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        # ---------- Target ----------
        self.problem_type_, self.num_classes_ = self._infer_problem_type(y)

        # Make sure labels are stable and serializable
        unique_labels = sorted(pd.unique(y.dropna()))
        # Cast to int when possible (common for binary 0/1). If not, keep as string.
        try:
            self.class_labels_ = [int(v) for v in unique_labels]
        except Exception:
            self.class_labels_ = [str(v) for v in unique_labels]

        self.label_to_index_ = {lab: i for i, lab in enumerate(self.class_labels_)}

        # Map y to indices robustly
        y_mapped = []
        for v in y:
            if pd.isna(v):
                y_mapped.append(np.nan)
            else:
                try:
                    vv = int(v)
                except Exception:
                    vv = str(v)
                y_mapped.append(self.label_to_index_[vv])
        y_index = pd.Series(y_mapped).astype(int).to_numpy()

        out_dim = 1 if self.problem_type_ == "binary" else self.num_classes_

        # ---------- Model ----------
        n_num_features = X_num.shape[1]
        self.model_ = self._build_fttransformer(
            n_num_features=n_num_features,
            cat_sizes=self.cat_sizes_,
            out_dim=out_dim,
        ).to(self.device_)

        loss_fn = nn.BCEWithLogitsLoss() if self.problem_type_ == "binary" else nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(p["lr"]),
            weight_decay=float(p["weight_decay"]),
        )

        # ---------- DataLoader ----------
        X_num_t = torch.tensor(X_num, dtype=torch.float32)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long)
        y_t = torch.tensor(y_index, dtype=torch.long)

        ds = TensorDataset(X_num_t, X_cat_t, y_t)
        dl = DataLoader(
            ds,
            batch_size=int(p["batch_size"]),
            shuffle=True,
            num_workers=int(p["num_workers"]),
            drop_last=False,
        )

        # ---------- Train loop ----------
        best_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for _epoch in range(int(p["epochs"])):
            self.model_.train()
            total_loss = 0.0
            n_seen = 0

            for xb_num, xb_cat, yb in dl:
                xb_num = xb_num.to(self.device_)
                xb_cat = xb_cat.to(self.device_)
                yb = yb.to(self.device_)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb_num, xb_cat)

                if self.problem_type_ == "binary":
                    yb_f = yb.float().view(-1, 1)  # BCE expects float targets shaped (n,1)
                    loss = loss_fn(logits, yb_f)
                else:
                    loss = loss_fn(logits, yb)

                loss.backward()
                optimizer.step()

                bs = yb.shape[0]
                total_loss += loss.item() * bs
                n_seen += bs

            epoch_loss = total_loss / max(1, n_seen)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(p["patience"]):
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        return self

    def _transform_X(self, X: pd.DataFrame):
        X = X[self.feature_columns_]

        # Numeric
        if len(self.num_cols_) > 0:
            X_num = self.num_imputer_.transform(X[self.num_cols_].astype("float32"))
            if self.num_mean_ is not None:
                X_num = (X_num - self.num_mean_) / self.num_std_
        else:
            X_num = np.zeros((len(X), 0), dtype=np.float32)

        # Categorical
        if len(self.cat_cols_) > 0:
            X_cat_list = []
            for col in self.cat_cols_:
                s_obj = X[col].astype("object")
                mapping = self.cat_maps_[col]

                encoded = np.zeros(len(s_obj), dtype=np.int64)
                mask = ~pd.isna(s_obj)
                if mask.any():
                    vals = pd.Series(s_obj[mask]).astype(str)
                    # unseen values -> 0
                    enc = vals.map(mapping).fillna(0).astype(np.int64).to_numpy()
                    encoded[mask.to_numpy()] = enc
                X_cat_list.append(encoded.reshape(-1, 1))

            X_cat = np.concatenate(X_cat_list, axis=1).astype(np.int64)
        else:
            X_cat = np.zeros((len(X), 0), dtype=np.int64)

        return X_num.astype(np.float32), X_cat

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        # FIXED indentation + stable behavior
        X_num, X_cat = self._transform_X(X)

        X_num_t = torch.tensor(X_num, dtype=torch.float32, device=self.device_)
        X_cat_t = torch.tensor(X_cat, dtype=torch.long, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_num_t, X_cat_t)

            if self.problem_type_ == "binary":
                # Return ONLY positive-class probability as 1D array: (n,)
                p1 = torch.sigmoid(logits).view(-1).cpu().numpy().astype(np.float32)
                return p1
            else:
                proba = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                return proba

    def save(self, path: str = None, verbose=True) -> str:
        path = super().save(path=path, verbose=verbose)

        torch.save(self.model_.state_dict(), os.path.join(path, "fttransformer.pt"))

        meta = {
            "feature_columns_": self.feature_columns_,
            "cat_cols_": self.cat_cols_,
            "num_cols_": self.num_cols_,
            "cat_maps_": self.cat_maps_,
            "cat_sizes_": self.cat_sizes_,
            "problem_type_": self.problem_type_,
            "num_classes_": self.num_classes_,
            "class_labels_": self.class_labels_,
            "num_mean_": None if self.num_mean_ is None else self.num_mean_.tolist(),
            "num_std_": None if self.num_std_ is None else self.num_std_.tolist(),
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, default=_json_safe)

        import joblib
        joblib.dump(self.num_imputer_, os.path.join(path, "num_imputer.pkl"))
        return path

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model = super().load(path=path, reset_paths=reset_paths, verbose=verbose)

        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        model.feature_columns_ = meta["feature_columns_"]
        model.cat_cols_ = meta["cat_cols_"]
        model.num_cols_ = meta["num_cols_"]
        model.cat_maps_ = meta["cat_maps_"]
        model.cat_sizes_ = meta["cat_sizes_"]
        model.problem_type_ = meta["problem_type_"]
        model.num_classes_ = meta["num_classes_"]
        model.class_labels_ = meta["class_labels_"]

        # Rebuild label mapping
        model.label_to_index_ = {lab: i for i, lab in enumerate(model.class_labels_)}

        model.num_mean_ = None if meta["num_mean_"] is None else np.array(meta["num_mean_"], dtype=np.float32)
        model.num_std_ = None if meta["num_std_"] is None else np.array(meta["num_std_"], dtype=np.float32)

        import joblib
        model.num_imputer_ = joblib.load(os.path.join(path, "num_imputer.pkl"))

        use_cuda = bool(getattr(model, "params", {}).get("use_cuda_if_available", False))
        model.device_ = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

        out_dim = 1 if model.problem_type_ == "binary" else model.num_classes_
        n_num_features = len(model.num_cols_)

        model.model_ = model._build_fttransformer(
            n_num_features=n_num_features,
            cat_sizes=model.cat_sizes_,
            out_dim=out_dim,
        ).to(model.device_)

        state = torch.load(os.path.join(path, "fttransformer.pt"), map_location="cpu")
        model.model_.load_state_dict(state)
        model.model_.eval()
        return model

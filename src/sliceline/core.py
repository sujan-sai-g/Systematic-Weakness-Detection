import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Dict, Any
import itertools


class SliceLineR:
    """
    Base SliceLine class without error correction.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sDims: list,
        lvl=2,
        alpha=0.95,
        err_column="error"  # Default error column name,
    ):
        self.alpha: float = alpha
        # Consider copying only the needed columns for improved memory efficiency.
        self.df: pd.DataFrame = data.copy()
        self.err_column: str = err_column
        self.sDims: List[str] = sDims
        self.true_series: pd.Series = pd.Series(True, index=self.df.index)
        self.global_err: float = self.df[self.err_column].sum()
        self._initialize_data()
        self.update_level(lvl)

    def _initialize_data(self):
        initial_values: List[Union[int, float]] = [0] + [0] * len(self.sDims)
        s, es = self.slice_statistics(self.true_series)
        global_score: float = self.sliceline_score(s, es)
        # Append computed statistics: n, es, eRate, and score.
        initial_values.extend(
            [s, es, es / s if s > 0 else np.nan, global_score]
        )
        columns: List[str] = (
            ["lvl"] + self.sDims + ["n", "es", "eRate", "score"]
        )
        self._scores: pd.DataFrame = pd.DataFrame(
            [initial_values], columns=columns
        )

    def sliceline_score(self, s, es):
        n = len(self.df)
        if s == 0 or self.global_err == 0:
            return np.nan
        e = self.global_err
        return self.alpha * (n / s * es / e - 1) - (1 - self.alpha) * (
            n / s - 1
        )

    @property
    def scores(self):
        return self._scores[self._scores["lvl"] <= self.lvl].copy()

    def level_indices(self, lvl: int):
        return self._get_sublevels(lvl, len(self.sDims))

    def _get_sublevels(self, items: int, length: int) -> List[List[int]]:
        """
        Recursively generate sign assignments for slicing dimensions.

        Args:
            items (int): Number of dimensions that should be non-zero.
            length (int): Total number of remaining dimensions.

        Returns:
            List[List[int]]: List of sign assignments.
        """
        if items > length:
            raise ValueError(
                "Number of non-zero items cannot exceed the total length."
            )

        # Special case: no non-zero entries
        if items == 0:
            return [[0] * length]

        results = []
        # Choose positions where non-zero values will be placed.
        for positions in itertools.combinations(range(length), items):
            # For each chosen combination, assign either +1 or -1.
            for signs in itertools.product([1, -1], repeat=items):
                row = [0] * length
                for pos, sign in zip(positions, signs):
                    row[pos] = sign
                results.append(row)
        return results

    def _update_data(self):
        df = self._scores
        for k in range(self.lvl + 1):
            cnt = len(df[df["lvl"] == k])
            if cnt == 0:
                self._calculate_stats_at_level(k)
        return

    def _calculate_stats_at_level(self, k: int):
        """
        Compute and update statistics for slices at a specific level.

        Args:
            k (int): The level for which to compute slice statistics.
        """
        sublevels: List[List[int]] = self.level_indices(k)
        num_extra_columns: int = self._scores.shape[1] - (len(self.sDims) + 1)
        # Create new rows with default statistics.
        new_rows: List[List[Union[int, float]]] = [
            [k] + sub + [0] * num_extra_columns for sub in sublevels
        ]
        new_df: pd.DataFrame = pd.DataFrame(
            new_rows, columns=self._scores.columns
        )
        self._scores = pd.concat([self._scores, new_df], ignore_index=True)
        for i in self._scores[self._scores["lvl"] == k].index:
            mask = self.slicing_mask(i)
            s, es = self.slice_statistics(mask)
            self._scores.loc[i, "n"] = s
            self._scores.loc[i, "es"] = es
            self._scores.loc[i, "eRate"] = es / s if s > 0 else np.nan
            self._scores.loc[i, "score"] = self.sliceline_score(s, es)
        return

    def slicing_mask(self, i: int):
        row: pd.Series = self._scores.loc[i]
        mask: pd.Series = self.true_series.copy()
        for dim in self.sDims:
            if row[dim] == 1:
                mask = mask & (self.df[dim])
            elif row[dim] == -1:
                mask = mask & (~self.df[dim])
        return mask

    def top_slices(self, k=10, sortBy="score"):
        res = self.scores
        amnt = min(len(res), k)
        res = res.sort_values(sortBy, ascending=False)
        return res.head(amnt)

    def update_level(self, lvl: int):
        if lvl == "max" or lvl == "all" or lvl > len(self.sDims):
            self.lvl = len(self.sDims)
        else:
            self.lvl = int(lvl)
        self._update_data()  # should have minimal overhead if lvl is decreased
        return self

    def slice_statistics(self, mask):
        df_sub: pd.DataFrame = self.df[mask]
        s: int = len(df_sub)
        es: float = df_sub[self.err_column].sum() if s > 0 else np.nan
        return s, es

    def update_alpha(self, alpha=1):
        self.alpha = alpha
        for idx in self._scores.index:
            s: int = self._scores.at[idx, "n"]
            es: float = self._scores.at[idx, "es"]
            self._scores.at[idx, "score"] = self.sliceline_score(s, es)
        return self


class SliceLineRplus(SliceLineR):
    """
    Enhanced SliceLine class with error estimation capabilities.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        sDims: list,
        errorEstimator: Union[None, callable] = None,
        errorEstimaterCounts: Union[None, callable] = None,
        lvl=2,
        alpha=0.95,
        err_column="error"  # Default error column name,
    ):
        self._eEstim = errorEstimator
        if errorEstimaterCounts is None:
            self._eEstimCounts = self._eEstim
            self._specialNestim = False
        else:
            self._eEstimCounts = errorEstimaterCounts
            self._specialNestim = True
        super().__init__(data, sDims, lvl, alpha, err_column)

    def _initialize_data(self):
        super()._initialize_data()
        self._scores["nC"] = float(self._scores["n"].iloc[0])
        self._scores["esC"] = float(self._scores["es"].iloc[0])
        self._scores["eRateC"] = self._scores["eRate"]
        self._scores["scoreC"] = 0.0

    def _semantic_data_groups(self, lvl: int) -> List[Dict[str, Any]]:
        """
        Group rows at a given level based on a normalized combination, where
        non-zero values are normalized to 1. Each group returns:
        - 'idx': the list of row indices that belong to the group,
        - 'combs': all variations for the representative combination.

        Args:
            lvl (int): The level at which to group the semantic data.

        Returns:
            List[Dict[str, Any]]: Grouping information for semantic slices.
        """
        # Filter rows at the given level.
        df_level = self._scores[self._scores["lvl"] == lvl]
        # Create a normalized version of the slicing dimensions:
        norm = df_level[self.sDims].map(lambda x: 0 if x == 0 else 1)
        df_level = df_level.assign(norm=norm.apply(tuple, axis=1))
        # Group rows by their normalized tuple.
        groups = df_level.groupby("norm")

        result = []
        for norm_key, group in groups:
            indices = group.index.tolist()
            # Use the first row as the representative combination.
            rep_comb = group.iloc[0][self.sDims].tolist()
            variations = self._build_variations(rep_comb)
            result.append({"idx": indices, "combs": variations})

        return result

    def _build_variations(self, lst: List[int]) -> List[List[int]]:
        """
        Generate all variations for a given combination list.
        Each 0 remains as 0; each non-zero is replaced by either +1 or -1.

        Args:
            lst (List[int]): The input combination.

        Returns:
            List[List[int]]: All variations of the combination.
        """
        # For each element, assign [0] if 0, else [1, -1].
        options = [[0] if x == 0 else [1, -1] for x in lst]
        # itertools.product generates the Cartesian product of options.
        return [list(prod) for prod in itertools.product(*options)]

    def update_alpha(self, alpha=1):
        super().update_alpha(alpha)
        df = self._scores
        for i in df.index:
            df.loc[i, "scoreC"] = self.sliceline_score(
                df.loc[i, "nC"], df.loc[i, "esC"]
            )
        return self

    def _calculate_stats_at_level(self, k: int):
        super()._calculate_stats_at_level(k)
        if self._eEstim is None:
            return
        pairings = self._semantic_data_groups(k)
        for pair in pairings:
            idx = pair["idx"]
            es = self._scores.loc[idx, "es"]
            ens = self._scores.loc[idx, "n"]
            ec, nc = self._eEstim(es, ens, pair["combs"])
            if self._specialNestim:
                _, nc = self._eEstimCounts(es, ens, pair["combs"])
            self._scores.loc[idx, "eRateC"] = ec
            self._scores.loc[idx, "nC"] = nc
            self._scores.loc[idx, "esC"] = nc * ec
        # Recompute scoreC for all rows at level k.
        for idx in self._scores[self._scores["lvl"] == k].index:
            nC = self._scores.at[idx, "nC"]
            esC = self._scores.at[idx, "esC"]
            self._scores.at[idx, "scoreC"] = self.sliceline_score(nC, esC)

    def top_slices(self, k=10, sortBy="scoreC"):
        return super().top_slices(k, sortBy)


class NaivePPIestimator:
    """
    Naive Prediction Powered Inference estimator for error correction.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        dims_clip: List[Any],
        clip_precisions: pd.DataFrame,
        label_map: Dict[Any, Any],
        inv_thresh: float = 0.1,
    ):
        self.clip_precisions = clip_precisions
        self.label_map = label_map
        self._nDims = len(dims_clip)
        self.inv_thresh = inv_thresh

        # Build error matrices and compute their determinants.
        self._eMats = [self._get_eMat(dim_clip) for dim_clip in dims_clip]
        self._dets = [np.linalg.det(mat) for mat in self._eMats]
        self._nMats = [self._get_nMat(dim_clip) for dim_clip in dims_clip]
        self.invertible = [abs(det) > self.inv_thresh for det in self._dets]
        self._iEMats = [
            np.linalg.inv(mat) if inv else np.identity(2)
            for mat, inv in zip(self._eMats, self.invertible)
        ]

    def _get_precision(self, dClip: Any) -> Tuple[float, float]:
        gt_column = [k for k, v in self.label_map.items() if v == dClip][0]
        p = self.clip_precisions[
            (self.clip_precisions["feature"] == gt_column)
            & (self.clip_precisions["value"] == 1)
        ]["precision"].item()
        ip = self.clip_precisions[
            (self.clip_precisions["feature"] == gt_column)
            & (self.clip_precisions["value"] == 0)
        ]["precision"].item()
        return p, ip

    def _get_eMat(self, dClip: Any) -> List[List[float]]:
        p, ip = self._get_precision(dClip)
        return [[ip, 1 - ip], [1 - p, p]]

    def _get_nMat(self, dClip: Any) -> List[List[float]]:
        p, ip = self._get_precision(dClip)
        return [[ip, 1 - p], [1 - ip, p]]

    def __call__(
        self, es: pd.Series, n: pd.Series, combs: List[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        e_vals = es.values
        n_vals = n.values
        e_rate = e_vals / n_vals

        iEmat = self._build_mat(combs, self._iEMats)
        Nmat = self._build_mat(combs, self._nMats)

        e_corr = iEmat @ e_rate
        n_corr = Nmat @ n_vals
        return e_corr, n_corr

    @staticmethod
    def _build_mat(
        combs: List[List[int]], mats: List[List[List[float]]]
    ) -> np.ndarray:
        n = len(combs)
        result = np.empty((n, n), dtype=float)

        for i, c1 in enumerate(combs):
            for j, c2 in enumerate(combs):
                value = 1.0
                # Loop over each dimension.
                for d in range(len(c1)):
                    # Only use the matrix's value if c1[d] is non-zero.
                    if c1[d] != 0:
                        index1 = (c1[d] + 1) // 2  # Maps -1 to 0 and 1 to 1.
                        index2 = (c2[d] + 1) // 2
                        value *= mats[d][index1][index2]
                result[i, j] = value

        return result
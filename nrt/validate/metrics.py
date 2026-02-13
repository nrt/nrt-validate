"""
Temporal accuracy metrics for monitoring algorithms.

This module implements frameworks for assessing how detection accuracy evolves
over time ("Time-to-Detection" analysis). It unifies methodologies from:

1.  **Bullock et al. (2022)**: "Timeliness in forest change monitoring..."
    - Implements *Initial Delay* and *Level-off* metrics using Sigmoid curves.
2.  **Tang et al. (2019)**: "Determining the optimal lag..."
    - Implements *Time-to-Accuracy* thresholds.
3.  **Pickens et al. (2025)**: "Rapid monitoring of global land change"
    - Supports strict temporal tolerance windows (e.g., +/- 30 days).

Methodology Note: Fixed vs. Dynamic Denominators
------------------------------------------------
A key challenge in temporal validation is defining the population of "Commission Errors"
(False Alarms) as a function of time lag.

* **Bullock et al. (2022) Codebase approach**: The reference notebook providede with the article filters
    predictions dynamically. If a prediction's lag is undefined (e.g., a False Alarm
    on a stable pixel), it is often excluded from the denominator at small lags.

* **nrt-validate approach (This Module)**: We enforce **Fixed Denominators** consistent with
    standard stratified accuracy assessment (Stehman 2014, Olofsson 2014).

    - **Producer's Accuracy (Recall)**: Denominator is *always* the total weighted population
      of Reference Events in the monitoring period.
    - **User's Accuracy (Precision)**: Denominator is *always* the total weighted population
      of Alerts issued in the monitoring period.

    **Implication**: A "Pure Commission Error" (Alert on a stable pixel) counts as a
    failure at *every* time lag. This prevents artificial inflation of accuracy at
    short lags and ensures the F1 curve asymptotes correctly.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Any, Dict
from kneed import KneeLocator
from scipy.stats import norm

from .estimators import BaseEstimator, SimpleRandomEstimator, StratifiedEstimator


class TemporalEvaluator:
    """Orchestrates the calculation of temporal accuracy metrics.

    This class decouples the **Temporal Logic** (is this alert within $L$ days
    of the event?) from the **Statistical Aggregation** (weighting, variances).

    It delegates the statistical heavy lifting to a provided
    :class:`~nrt.validate.estimators.BaseEstimator` instance.

    Attributes:
        estimator (BaseEstimator): Strategy for statistical aggregation (SRS or Stratified).
        y_true (np.ndarray): Reference dates (days since epoch). 0 indicates No Change.
        y_pred (np.ndarray): Detected dates (days since epoch). 0 indicates No Detection.
        diffs (np.ndarray): Pre-calculated temporal differences (Pred - True).
        ref_exists (np.ndarray): Boolean mask of all reference events (Fixed PA Denominator).
        pred_exists (np.ndarray): Boolean mask of all detections (Fixed UA Denominator).
    """
    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 estimator: Optional[BaseEstimator] = None,
                 experiment_start: int = 0,
                 experiment_end: int = 99999):
        """Initialize the evaluator with temporal bounds.

        Args:
            y_true: Array of reference dates (integers). 0 = No Change.
            y_pred: Array of detection dates (integers). 0 = No Detection.
            estimator: Instance of BaseEstimator. Defaults to SimpleRandomEstimator.
            experiment_start: Integer date. Events/Alerts before this are ignored (masked to 0).
            experiment_end: Integer date. Events/Alerts after this are ignored (masked to 0).

        Note:
            Applying `experiment_start` and `experiment_end` is crucial to correctly
            classify "Out of Range" errors as Omissions/Commissions rather than
            Late Detections.
        """
        self.estimator = estimator if estimator is not None else SimpleRandomEstimator()
        y_true = np.array(y_true, dtype=np.int64)
        y_pred = np.array(y_pred, dtype=np.int64)

        # 1. Apply Experiment Temporal Bounds
        # Events outside bounds are treated as "Stable" (0) for the purpose of this experiment
        self.y_true = np.where((y_true >= experiment_start) & (y_true <= experiment_end), y_true, 0)
        self.y_pred = np.where((y_pred >= experiment_start) & (y_pred <= experiment_end), y_pred, 0)

        # 2. Pre-calculate Populations (The "Fixed Denominators")
        self.ref_exists = (self.y_true > 0)
        self.pred_exists = (self.y_pred > 0)

        # 3. Pre-calculate Differences
        self.both_exist = self.ref_exists & self.pred_exists
        # Initialize with Infinity so non-matches never satisfy "diff <= lag"
        self.diffs = np.full_like(self.y_true, np.inf, dtype=np.float64)

        # diff = pred - true
        # Negative = Anticipation (Alert before Event)
        # Positive = Delay (Alert after Event)
        self.diffs[self.both_exist] = (
            self.y_pred[self.both_exist] - self.y_true[self.both_exist]
        )

        # Cache strata indices for bootstrapping speed
        if hasattr(self.estimator, 'strata_labels'):
            self._strata_indices = {
                s: np.where(self.estimator.strata_labels == s)[0]
                for s in np.unique(self.estimator.strata_labels)
            }
        else:
            self._strata_indices = None

    def accuracy_at_lag(self,
                        lag: int,
                        metric: str = 'f1',
                        negative_tolerance: int = 30) -> float:
        """Computes a point estimate of accuracy for a specific temporal lag window.

        A sample is considered a **True Positive (TP)** at lag $L$ if and only if:

        1.  A Reference Event exists inside the experiment bounds.
        2.  A Detection exists inside the experiment bounds.
        3.  The temporal difference $D = T_{pred} - T_{ref}$ satisfies:
            $$-Tolerance_{neg} \le D \le Lag$$

        Scenarios handled:
            - **Pure Commission**: Ref=0, Pred=1. $D=\inf$. Result: Never TP.
              Always in UA Denom. Correctly lowers UA.
            - **Pure Omission**: Ref=1, Pred=0. $D=\inf$. Result: Never TP.
              Always in PA Denom. Correctly lowers PA.
            - **Late Detection**: Ref=1, Pred=1, $D > Lag$. Result: Not TP (yet).
              Counts as error for both UA and PA at this specific lag.

        Args:
            lag: Maximum allowed delay (days) for a detection to be considered correct.
            metric: 'ua' (User's), 'pa' (Producer's), or 'f1'.
            negative_tolerance: Maximum allowed anticipation (days).

        Returns:
            Float value of the metric (0.0 to 1.0).
        """
        # Define Hit Condition (True Positive)
        is_match = (self.diffs >= -negative_tolerance) & (self.diffs <= lag)
        tp_mask = self.both_exist & is_match

        if metric.lower() == 'pa':
            val, _ = self.estimator.estimate_ratio(tp_mask, self.ref_exists)
            return val
        elif metric.lower() == 'ua':
            val, _ = self.estimator.estimate_ratio(tp_mask, self.pred_exists)
            return val
        elif metric.lower() == 'f1':
            ua, _ = self.estimator.estimate_ratio(tp_mask, self.pred_exists)
            pa, _ = self.estimator.estimate_ratio(tp_mask, self.ref_exists)
            if (ua + pa) == 0:
                return 0.0
            return 2 * (ua * pa) / (ua + pa)

        return 0.0

    def compute_curve(self,
                      lags: List[int],
                      metrics: List[str] = ['f1'],
                      negative_tolerance: int = 30) -> pd.DataFrame:
        """Calculates accuracy metrics across a range of lags to build the 'Sigmoid' curve.

        Args:
            lags: List of days to evaluate (e.g. range(0, 180, 5)).
            metrics: List of metrics to compute ['ua', 'pa', 'f1'].
            negative_tolerance: Anticipation window.

        Returns:
            pd.DataFrame: DataFrame with columns ['lag', 'metric', 'value'].
        """
        data = []
        for lag in lags:
            # We construct masks manually here for slight efficiency gain
            # if we wanted to compute multiple metrics on one mask,
            # but relying on accuracy_at_lag keeps logic DRY.
            row = {'lag': lag}

            # Cache intermediate results to avoid re-computing TP mask for F1
            is_match = (self.diffs >= -negative_tolerance) & (self.diffs <= lag)
            tp_mask = self.both_exist & is_match

            # Compute UA/PA once if needed
            ua, pa = 0.0, 0.0
            if 'ua' in metrics or 'f1' in metrics:
                ua, _ = self.estimator.estimate_ratio(tp_mask, self.pred_exists)
                if 'ua' in metrics: row['ua'] = ua

            if 'pa' in metrics or 'f1' in metrics:
                pa, _ = self.estimator.estimate_ratio(tp_mask, self.ref_exists)
                if 'pa' in metrics: row['pa'] = pa

            if 'f1' in metrics:
                if (ua + pa) > 0:
                    row['f1'] = 2 * (ua * pa) / (ua + pa)
                else:
                    row['f1'] = 0.0

            data.append(row)

        return pd.DataFrame(data)

    def find_initial_delay(self,
                           df_curve: pd.DataFrame,
                           metric: str = 'pa',
                           threshold: float = 0.02) -> Tuple[int, float]:
        """Finds the 'Initial Delay' metric (Bullock et al. 2022).

        The Initial Delay is the minimum lag required to produce a non-negligible
        amount of correct detections. Bullock et al. define this as the point
        where Omission Error drops below 98% (i.e., Producer's Accuracy > 2%).

        Args:
            df_curve: Result from `compute_curve`.
            metric: Metric to evaluate (default 'pa').
            threshold: Value to exceed (default 0.02).

        Returns:
            (lag, value): The lag day and metric value at the crossing point.
                          Returns (NaN, NaN) if threshold is never reached.
        """
        hits = df_curve[df_curve[metric] > threshold]

        if hits.empty:
            return np.nan, np.nan

        first = hits.iloc[0]
        return int(first['lag']), first[metric]

    def find_level_off(self,
                       df_curve: pd.DataFrame,
                       metric: str = 'f1',
                       start_lag: int = 0) -> Tuple[int, float]:
        """Finds the 'Level Off Point' using the Kneedle algorithm (Bullock et al. 2022).

        The Level Off Point represents the lag where accuracy effectively saturates,
        indicating the "reasonable" time to wait for a high-quality map.

        Args:
            df_curve: Result from `compute_curve`.
            metric: Metric to evaluate (usually 'f1').
            start_lag: Optimization constraint. Only search for the knee *after* this lag (typically after the Initial Delay).

        Returns:
            (lag, value): The lag day and metric value at the knee.
                          Returns (NaN, NaN) if no knee is found.
        """
        subset = df_curve[df_curve['lag'] >= start_lag].copy()

        # Kneedle requires at least a few points
        if len(subset) < 3:
            return np.nan, np.nan

        kneedle = KneeLocator(
            subset['lag'].values,
            subset[metric].values,
            curve='concave',
            direction='increasing'
        )

        if kneedle.knee is None:
            return np.nan, np.nan

        # Retrieve the Metric Value (Y) at the Knee Lag (X)
        knee_y = subset.loc[subset['lag'] == kneedle.knee, metric].values[0]
        return int(kneedle.knee), knee_y

    def time_to_accuracy(self,
                         df_curve: pd.DataFrame,
                         target: float = 0.90,
                         metric: str = 'ua') -> int:
        """Calculates the 'Time to Accuracy' metric (Tang et al. 2019).

        Finds the minimum lag required to reach a specific accuracy target.

        Args:
            df_curve: Result from `compute_curve`.
            target: Target accuracy value (e.g. 0.90).
            metric: Metric to evaluate.

        Returns:
            int: The lag in days, or -1 if the target is never reached.
        """
        hits = df_curve[df_curve[metric] >= target]
        if hits.empty:
            return -1
        return int(hits.iloc[0]['lag'])

    def bootstrap_metrics(self,
                          lags: List[int],
                          n_boot: int = 1000,
                          negative_tolerance: int = 30,
                          initial_delay_metric: str = 'pa',
                          initial_delay_thresh: float = 0.02) -> Dict[str, Any]:
        """Estimates uncertainty for curve-derived metrics using Stratified Bootstrapping.

        Unlike standard accuracy bootstrapping (which bootstraps the metric itself),
        this method bootstraps the *entire curve construction* to derive the
        standard error of the **Initial Delay** and **Level Off Point**.

        Methodology:
        1.  Resample the dataset (stratified by original design).
        2.  Re-calculate the full accuracy curve for the resample.
        3.  Re-calculate Initial Delay and Level Off for that curve.
        4.  Repeat `n_boot` times to build sampling distributions.

        Args:
            lags: Search space for the curve.
            n_boot: Number of bootstrap iterations.
            negative_tolerance: Anticipation window.
            initial_delay_metric: Metric for initial delay ('pa' or 'f1').
            initial_delay_thresh: Threshold for initial delay.

        Returns:
            Dict: Contains statistics (mean, se, 95% CI) for:
                  'initial_delay': {lag_mean, lag_se, val_mean, val_se...}
                  'level_off': {lag_mean, lag_se, val_mean, val_se...}
        """
        # --- Internal Vectorized Bootstrap Engine ---
        # To make 1000 curve calculations feasible, we bypass the pandas/object overhead
        # and use pure numpy matrix operations.

        # 1. Prepare Storage
        dist_id_lag = [] # Initial Delay Lag
        dist_id_val = [] # Initial Delay Value
        dist_lo_lag = [] # Level Off Lag
        dist_lo_val = [] # Level Off Value

        lags_arr = np.array(lags)

        # 2. Get Base Weights (Static)
        base_weights = getattr(self.estimator, 'sample_weights', np.ones_like(self.y_true, dtype=float))

        for _ in range(n_boot):
            # A. Stratified Resampling of Indices
            indices = self._get_resampled_indices()

            # B. Slice Data (Data + Weights must move together)
            res_diffs = self.diffs[indices]
            res_ref_ex = self.ref_exists[indices]
            res_pred_ex = self.pred_exists[indices]
            res_weights = base_weights[indices]

            # C. Vectorized Curve Calculation
            # Expand diffs to (N, 1) and compare with lags (1, L) -> (N, L) boolean matrix
            is_match = (res_diffs[:, None] >= -negative_tolerance) & (res_diffs[:, None] <= lags_arr[None, :])

            # Intersection (TP) Matrix: (N, L)
            # Must exist in both Ref and Pred AND be in temporal window
            both_ex = (res_ref_ex & res_pred_ex)[:, None]
            tp_matrix = both_ex & is_match

            # Weighted Counts (Dot Product)
            # (N,) dot (N, L) -> (L,)
            tp_counts = np.dot(res_weights, tp_matrix)
            ref_total = np.dot(res_weights, res_ref_ex)
            pred_total = np.dot(res_weights, res_pred_ex)

            # Metric Vectors (L,)
            pa_vec = np.divide(tp_counts, ref_total, out=np.zeros_like(tp_counts), where=ref_total!=0)
            ua_vec = np.divide(tp_counts, pred_total, out=np.zeros_like(tp_counts), where=pred_total!=0)

            sum_vec = ua_vec + pa_vec
            f1_vec = np.divide(2 * ua_vec * pa_vec, sum_vec, out=np.zeros_like(sum_vec), where=sum_vec!=0)

            # D. Derive Metrics from Vectors

            # --- Initial Delay ---
            target_vec = pa_vec if initial_delay_metric == 'pa' else f1_vec
            idx_candidates = np.where(target_vec > initial_delay_thresh)[0]

            if len(idx_candidates) > 0:
                id_idx = idx_candidates[0]
                # Store results
                dist_id_lag.append(lags_arr[id_idx])
                dist_id_val.append(f1_vec[id_idx]) # Bullock uses F1 value at the PA delay point

                # --- Level Off (Knee) ---
                # Search only after initial delay
                knee_x = lags_arr[id_idx:]
                knee_y = f1_vec[id_idx:]

                if len(knee_x) > 2:
                    kneedle = KneeLocator(knee_x, knee_y, curve='concave', direction='increasing')
                    if kneedle.knee is not None:
                        dist_lo_lag.append(kneedle.knee)

                        # Find corresponding F1 value
                        # kneedle.knee is an X value (Lag). Find index in original array.
                        k_idx = np.searchsorted(lags_arr, kneedle.knee)
                        if k_idx < len(lags_arr):
                            dist_lo_val.append(f1_vec[k_idx])

        # 3. Summarize Distributions
        return {
            'initial_delay': self._summarize_dist(dist_id_lag, dist_id_val),
            'level_off': self._summarize_dist(dist_lo_lag, dist_lo_val)
        }

    def _get_resampled_indices(self):
        """Generates a list of indices resampled with replacement within strata."""
        if self._strata_indices is not None:
            indices = []
            for idxs in self._strata_indices.values():
                indices.append(np.random.choice(idxs, size=len(idxs), replace=True))
            return np.concatenate(indices)
        else:
            # Simple Random fallback
            n = len(self.y_true)
            return np.random.choice(n, size=n, replace=True)

    def _summarize_dist(self, lags, vals):
        """Calculates statistics (Mean, SE, CI) for metric distributions."""
        if not lags:
            return {'lag_mean': np.nan}

        l_arr = np.array(lags)
        v_arr = np.array(vals)

        return {
            'lag_mean': np.mean(l_arr),
            'lag_se': np.std(l_arr, ddof=1),
            'lag_ci': np.percentile(l_arr, [2.5, 97.5]),
            'val_mean': np.mean(v_arr),
            'val_se': np.std(v_arr, ddof=1),
            'val_ci': np.percentile(v_arr, [2.5, 97.5])
        }

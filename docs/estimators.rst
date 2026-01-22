Estimators
==========

The ``nrt.validate.estimators`` module provides statistically rigorous tools for estimating map accuracy (User's, Producer's, Overall) and area proportions from sample data.

.. note::
    This module implements the generalized estimators described in **Stehman (2014)**. This formulation provides the **exact analytical solution** for stratified designs and covers the popular "Good Practices" workflow described in **Olofsson et al. (2014)** as a special case.

Theory: Stehman vs. Olofsson
----------------------------

In the remote sensing literature, two papers are frequently cited for accuracy assessment:

1. **Olofsson et al. (2014)**: "Good practices for estimating area and assessing accuracy of land change".
   This paper focuses on the most common scenario where the **Strata** used for sampling are identical to the **Map Classes** being evaluated. This is often called "Post-Stratification" or standard Stratified Random Sampling.

2. **Stehman (2014)**: "Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes".
   This paper presents the **generalized** formulas. It handles the complex case where your sampling strata do *not* match your map classes (e.g., using "buffer zones" as strata to validate a binary change map).

**Key Takeaway:**
The ``StratifiedEstimator`` class in this package implements the generalized **Stehman (2014)** formulas. If you pass it data where ``strata_labels == map_labels``, it mathematically simplifies to the exact results found in **Olofsson et al. (2014)**. You do not need a separate estimator for the Olofsson method.


Bootstrapping
-------------

While metrics like Overall Accuracy, User's Accuracy, and Producer's Accuracy have known analytical variance estimators (derived in Stehman, 2014), others do not.

The **F1 Score** is the harmonic mean of User's and Producer's accuracy. Because it is a non-linear combination of two ratio estimators, there is no simple analytical solution for its Standard Error (SE). To address this, the module provides a **Bootstrap** implementation (Stratified Resampling) to estimate the uncertainty of the F1 Score. This follows the methodology used in recent timeliness assessment frameworks (e.g., **Bullock et al., 2022**).

Usage Examples
--------------

Example 1: The General Case (Stehman 2014)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example reproduces the numerical results from the Stehman (2014) manuscript (Section 4), where the strata (A, B, C, D) do not perfectly align with the map or reference classes.

.. code-block:: python

    import numpy as np
    from nrt.validate.estimators import StratifiedEstimator

    # 1. Define Sampling Design (Strata)
    # 4 Strata with n=10 samples each (Total n=40)
    strata_labels = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)

    # Population sizes (Nh) for each stratum (Total Area)
    pop_sizes = {1: 40000, 2: 30000, 3: 20000, 4: 10000}

    # 2. Define Observed Data
    # Map Labels (m) and Reference Labels (r)
    # Note: These vectors exactly match the paper's numerical example
    map_labels = np.array(
        ["A"]*7 + ["B"]*3 +             # Stratum 1
        ["A"] + ["B"]*9 +               # Stratum 2
        ["B"]*4 + ["C"]*6 +             # Stratum 3
        ["D"]*10                        # Stratum 4
    )

    ref_labels = np.array(
        ["A"]*5 + ["C", "B", "A", "B", "C"] +       # Stratum 1
        ["A"] + ["B"]*5 + ["A", "A", "B", "B"] +    # Stratum 2
        ["C"]*2 + ["B", "A"] + ["C"]*3 + ["D"]*2 + ["B"] + # Stratum 3
        ["D"]*7 + ["C", "C", "B"]                   # Stratum 4
    )

    # 3. Initialize Estimator
    est = StratifiedEstimator(strata_labels, pop_sizes)

    # 4. Compute Metrics
    # Overall Accuracy (OA)
    oa, oa_se = est.overall_accuracy(ref_labels, map_labels)
    print(f"OA: {oa:.2f} (SE: {oa_se:.3f})")
    # Expected: OA: 0.63 (SE: 0.085)

    # User's Accuracy for Class 'B'
    ua_b, ua_b_se = est.user_accuracy(ref_labels, map_labels, label="B")
    print(f"UA (B): {ua_b:.3f} (SE: {ua_b_se:.3f})")
    # Expected: UA (B): 0.574 (SE: 0.125)

Example 2: The Special Case (Olofsson et al. 2014)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example reproduces the "Good Practices" example from Section 5 of Olofsson et al. (2014). Here, the sampling strata are the map classes themselves.

.. code-block:: python

    import numpy as np
    from nrt.validate.estimators import StratifiedEstimator

    # Data from Olofsson et al. (2014) Table 8
    # 1: Deforestation, 2: Forest Gain, 3: Stable Forest, 4: Stable Non-Forest

    # Population Sizes (Pixels)
    Nh = {1: 200_000, 2: 150_000, 3: 3_200_000, 4: 6_450_000}

    # Reconstruct the sample vectors from the paper's confusion matrix
    # Format: (stratum_id, count_ref_1, count_ref_2, count_ref_3, count_ref_4)
    matrix_rows = [
        (1, 66, 0, 5, 4),    # Stratum 1 (Map Class 1)
        (2, 0, 55, 8, 12),   # Stratum 2 (Map Class 2)
        (3, 1, 0, 153, 11),  # Stratum 3 (Map Class 3)
        (4, 2, 1, 9, 313)    # Stratum 4 (Map Class 4)
    ]

    s_list, r_list = [], []
    for stratum, *counts in matrix_rows:
        for ref_class_idx, count in enumerate(counts, 1):
            s_list.extend([stratum] * count)
            r_list.extend([ref_class_idx] * count)

    s_arr = np.array(s_list)
    r_arr = np.array(r_list)
    # In this design, Map Labels == Strata Labels
    m_arr = s_arr

    # Initialize
    est = StratifiedEstimator(s_arr, Nh)

    # Compute Estimates (Section 5.2.1 of Paper)
    # Note: We use integer label=1 because our data arrays are integers
    ua_def, ua_se = est.user_accuracy(r_arr, m_arr, label=1)
    print(f"UA Deforestation: {ua_def:.2f} (SE: {ua_se:.2f})")
    # Expected: 0.88 (SE: ~0.04)

    # Area Estimate (Section 5.2.2)
    # Area = Proportion * Total Map Area
    prop_def, prop_se = est.estimate_mean(r_arr == 1)
    area_def_ha = prop_def * 10_000_000 * 0.09 # 0.09 ha per pixel
    margin = (prop_se * 1.96) * 10_000_000 * 0.09

    print(f"Area Deforestation: {int(area_def_ha)} +/- {int(margin)} ha")
    # Expected: 21,158 +/- 6158 ha

Example 3: Bootstrapping for F1 Score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to compute the **F1 Score** and its standard error using bootstrapping. This is necessary because there is no simple analytical formula for the variance of the F1 score (harmonic mean of UA and PA).

We use the same Olofsson dataset as Example 2.

.. code-block:: python

    # ... (Assuming data setup from Example 2 is already run) ...

    # Compute F1 for 'Deforestation' (Class 1)
    # se_method='simple_bootstrap' triggers stratified resampling
    f1, f1_se = est.f1_score(
        y_true=r_arr,
        y_pred=m_arr,
        label=1,
        se_method='simple_bootstrap',
        n_boot=1000
    )

    print(f"F1 Score (Deforestation): {f1:.3f} +/- {f1_se * 1.96:.3f}")
    # Expected: ~0.81 +/- ~0.13

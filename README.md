# Signatures Induced by Network Geometry

---

## Overview

This repository studies how the *geometry underlying a network* induces stable, low-dimensional structure in the space of network observables.

The guiding question of this work is:

**Can the geometry of a network be inferred from collective patterns of observables, even when individual observables and pairwise correlations are unstable under parameter variation?**

Rather than searching for invariant observables or stable correlations, this work adopts a **collective, geometric, and spectral perspective**. The central claim is that geometric information is not encoded in individual metrics, but instead emerges in the *dominant collective modes* of observable correlations.

---

## Why Standard Stability Arguments Fail

A common assumption in network science is that if a structural property is fundamental, it should appear as either:
1. a stable observable, or
2. a stable pairwise correlation between observables.

This assumption is mathematically fragile in geometric network models.

In hyperbolic networks, variations in parameters such as temperature, average degree, power-law exponent, or system size naturally induce large fluctuations in degree distributions, clustering coefficients, path lengths, and centrality measures. Expecting stability at the level of individual observables in such settings is therefore unjustified.

This motivates a shift in viewpoint: **instead of asking which observables are stable, we ask whether the space spanned by observables exhibits stable structure**.

---

## Observable Space Representation

Each network realization is represented as a vector in a high-dimensional observable space:

$$
\mathbf{x} = (x_1, x_2, \dots, x_d)
$$

where each coordinate corresponds to a network observable.

As model parameters are varied, these vectors form a cloud of points in $d$-dimensional space. Each point corresponds to one network realization.

---

## The Cloud Analogy

The key geometric insight is that parameter changes do not move this cloud arbitrarily.

Instead, the cloud deforms **anisotropically**:
- it stretches strongly along certain directions,
- while remaining constrained along others.

These preferred directions encode geometric information.

The task is therefore not to track individual coordinates, but to **identify the dominant directions along which the cloud deforms**.

---

## Correlation Matrices as Shape Descriptors

To characterize the shape of the observable cloud, correlation matrices are constructed across observables.

Given observable vectors $x_i$, the correlation matrix $C$ is defined as:

$$
C_{ij} = \mathrm{corr}(x_i, x_j)
$$

Individual entries of $C$ are allowed to vary significantly across parameter slices. What matters is not the stability of individual correlations, but the **global structure** of the matrix.

Geometrically, $C$ captures the second-order shape of the observable cloud.

---

## Spectral Decomposition and Collective Modes

The spectral decomposition of the correlation matrix,

$$
C = V \Lambda V^{\top}
$$

reveals the intrinsic geometry of the observable space.

Here:
- the eigenvalues $\lambda_k$ quantify variance along principal directions,
- the eigenvectors $v_k$ define collective modes involving multiple observables.

A dominant leading eigenvalue indicates that the cloud is effectively low-dimensional, with most variance concentrated along a small number of directions.

---

## Interpretation of Leading Eigenvectors

The components of the leading eigenvector,

$$
v_1 = (v_{11}, v_{12}, \dots, v_{1d}),
$$

measure the participation of each observable in the dominant collective mode.

Observables with large absolute components contribute most strongly to the geometry-induced deformation of the cloud.

These observables are *emergent*: they are not stable individually, but they consistently dominate the collective mode.

---

## Pairwise Instability Is Expected

Pairwise correlation instability is not a failure mode. It is expected.

As parameters vary, the observable cloud undergoes nonlinear deformation. This naturally induces large fluctuations in individual entries of $C$.

The correct question is therefore not whether $C_{ij}$ is stable, but whether the **dominant eigenspaces of $C$ remain stable**.

---

## Why Eigenvalue and SVD Non-Convergence Occurs

During analysis, numerical routines such as eigenvalue decomposition or SVD may fail to converge.

This behavior arises when:
- the matrix is ill-conditioned,
- observables are strongly collinear,
- variance is concentrated in a narrow subspace.

Geometrically, this corresponds to a cloud that is extremely elongated along a few directions.

Importantly, this is not numerical noise. It is a signature of **strong geometric constraint**.

When mild regularization restores convergence without destroying dominant eigenstructure, it confirms that the observed low-dimensionality is intrinsic.

---

## Regularization as a Diagnostic Tool

Regularization is used not to force results, but to probe stability.

If regularization preserves:
- leading eigenvalues,
- leading eigenvectors,
- dominant observable rankings,

then the collective structure is robust.

In contrast, if regularization destroys these features, the structure is likely spurious.

---

## Emergent Observables and Geometric Signatures

Averaging eigenvector contributions across parameter slices yields a ranked list of observables that consistently dominate the collective mode.

In hyperbolic networks, these typically include:

$$
\{\text{core-periphery},\ \text{k-core depth},\ \text{average degree},\ \text{degree variance},\ \text{clustering-related measures}\}
$$

These observables jointly define a **geometric signature**.

No single observable is sufficient. The signature exists only at the collective level.

---

## Verification Logic

The analysis follows a strict logical sequence:

1. Show that pairwise correlations are unstable.
2. Show that leading eigenvalues are comparatively stable.
3. Show that leading eigenvectors have high cosine similarity across slices.
4. Show that the same observables dominate the collective mode.

Only when all four conditions are satisfied do we claim a robust geometric signature.

---

## Final Interpretation

The results demonstrate that network geometry does not manifest as stable local measurements.

Instead, geometry imposes global constraints on how observables fluctuate together.

These constraints appear as low-dimensional collective modes that persist even when individual correlations fail.

---

## Summary

This work establishes that:
- instability at the pairwise level does not imply absence of structure,
- geometry constrains observable space collectively,
- spectral methods are essential for extracting geometric information,
- hyperbolic networks exhibit robust collective signatures.

The framework is general and applicable to other geometric and non-geometric network models.


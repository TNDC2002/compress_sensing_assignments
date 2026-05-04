# Assignment 4 Report: Sparse Image Recovery

## 1) Experimental setup

- Test images: `Phantom` (synthetic), `Brain` (real), `Boat` (real).
- Patch size: `8 x 8` (`n = 64` pixels per patch).
- Sparsifying transform: 2D DCT (equivalently sparse coefficients in IDCT basis).
- Recovery model: patch-wise LASSO solved by ISTA (chosen because sparsity level `S` is unknown).
- Sensing schemes:
  - `Random Gaussian`: dense random sensing matrix.
  - `Random Subsampling`: random selection of pixels in the patch domain.
- Measurement counts per patch: `M in {6, 13, 19, 26, 32, 38}` (from ratios `0.1` to `0.6` of 64).
- Distortion metric:
  - `MSE = (1 / N^2) * ||x_hat - x||_2^2`
  - `PSNR = 10 * log10(MAX^2 / MSE)`, with `MAX = 255`.

## 2) PSNR vs measurements

Observed from the generated CSV files and PSNR plots:

- PSNR increases monotonically with `M` for all images and both sensing schemes.
- `Random Gaussian` gives consistently higher PSNR than `Random Subsampling` at every tested `M`.
- The quality gap is larger at low `M` and becomes smaller at higher `M`.

Final PSNR at `M = 38`:

- `Phantom`: Gaussian `10.83 dB`, Subsampling `10.44 dB`
- `Brain`: Gaussian `15.28 dB`, Subsampling `15.08 dB`
- `Boat`: Gaussian `9.72 dB`, Subsampling `9.44 dB`

## 3) Which sensing scheme is better?

Based on reconstruction quality (PSNR), **Random Gaussian is better** in this experiment:

- It outperforms random subsampling at all tested measurement levels for all three images.
- This is consistent with compressed sensing theory: Gaussian sensing is highly incoherent with DCT/IDCT-type sparsifying bases.

## 4) Which sensing scheme is faster?

From the recorded runtime in your full run, **Random Gaussian is also faster on average** here:

- `Phantom`: Gaussian faster at every `M`.
- `Brain`: Gaussian faster at most `M` values (very close at `M=32`).
- `Boat`: Gaussian much faster at all `M` values.

So for this implementation, Gaussian is both more accurate and faster.

## 5) How many measurements are needed for reasonable recovery?

In this run, absolute PSNR values remain modest (all below `16 dB`) due to strict global image reconstruction and current solver settings. Therefore:

- Within tested range (`M <= 38`, i.e., up to `60%` of patch dimension), recovery quality still improves steadily but has not reached high-fidelity visual quality.
- A practical conclusion is that **more than 38 measurements per 8x8 patch** (or stronger/tuned recovery settings) are needed for clearly better visual reconstruction, especially for `Boat` and `Phantom`.

Relative recommendation from your curves:

- `Brain` becomes the most usable first (best PSNR among three images).
- `Boat` and `Phantom` need larger `M` than tested for good quality.

## 6) Approximate sparsity level of each image

Average patch sparsity was estimated from DCT coefficients using threshold `|coef| > eps` (`eps = 3.0`):

- `Phantom`: `~11.01 / 64`
- `Brain`: `~38.88 / 64`
- `Boat`: `~25.21 / 64`

Interpretation:

- `Phantom` is the sparsest in DCT domain.
- `Brain` is the least sparse (largest effective support).
- `Boat` is intermediate.

## 7) Key observations for best recovery performance

- Increase measurement count `M`: this is the strongest driver of PSNR improvement.
- Prefer `Random Gaussian` sensing with IDCT/DCT sparsity.
- Use patch-wise transform sparsity (8x8 is valid and computationally manageable).
- Tune recovery hyperparameters (`lambda`, ISTA iterations), and optionally test `16x16` patches for a quality/runtime trade-off.

## 8) Conclusion

This assignment confirms the feasibility of compressed sensing on real images with unknown sparsity:

- PSNR improves as measurements increase.
- Random Gaussian sensing performs better than random pixel subsampling in both quality and speed for this implementation.
- Effective sparsity differs significantly across images, which directly impacts reconstruction difficulty and required number of measurements.
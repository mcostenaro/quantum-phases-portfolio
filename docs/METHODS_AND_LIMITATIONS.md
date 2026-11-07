# Methods and limitations

## Scientific question and design

Does adding ten synthetic examples per phase improve classification from
2, 3, 4, 5 or 6 real training examples per phase, and does a trained generator
outperform simple controls given the same information and sample count?

The main datasets describe three spin-1 Hamiltonians: H1 (XXZ with single-ion
anisotropy), H2 (bond-alternating XXZ), and H3 (bilinear–biquadratic). The
research representation is 21 local spin correlators: x/y/z components at
distances 1–7 on an L=12 ring. There is no aggregation with DMRG representations.

For each (dataset, training size, seed), sample real training points by phase
without replacement; all remaining points form the complementary test set.
Train one GAN or DDPM separately for each phase, using only that real training
set. Compare real-only and augmented 1-NN on exactly the same test points.
Normalize each input vector to L2 norm; compute Euclidean distances directly
in float64. Exact distance ties use training order, with real points first.

The 100 main seeds are separate from 30 tuning seeds. Training sets are nested
across sizes for a fixed seed, and complementary tests overlap across seeds.
These repetitions are not independent physical experiments.

## Arms and frozen settings

| Arm | Definition |
|---|---|
| Real | Only sampled real training points |
| Copies | Repeat real points cyclically; zero gain in 1-NN is an integrity check |
| Box center | Repeat the coordinatewise (minimum + maximum)/2 point |
| Interpolation | Convex combination of two distinct same-phase training points |
| Jitter | Training point plus Gaussian noise, sigma=0.005, clipped to [−1,1] |
| GAN v3 | Dense VanillaGAN per phase, physical scale, tanh output, width 64, Adam lr=0.0002, 3,000 steps |
| DDPM v3 | Per-phase denoiser predicting x0, cosine schedule with 100 noise steps, posterior variance, lr=0.001, no dropout; 10,000 steps H1, 3,000 H2/H3 |

All augmentation arms add ten points per phase. Physical scaling is fixed,
not fitted to the small training sample. The box center is generally different
from the arithmetic mean. Convex interpolation of correlations can describe
a mixed state; it does not imply a ground state of the original Hamiltonian.
Bounds |C| <= 1 alone do not certify a physically realizable quantum state.

Tuning left one dataset out: select each dataset's setting using the other two
and disjoint tuning seeds. Admissibility included evidence of learning and
minimal physical-validity checks. GAN had 8 candidates. DDPM initially had no
admissible candidate; 5 candidates were added later, for 13 total, leading to
the x0 settings. The unequal and partly adaptive tuning budget limits fairness
claims. Settings were frozen before the main grid and were not retuned after
excluding seven anomalous rows, although tuning had included those rows.

## Criterion and interpretation

Primary metric: macro-F1. Also report balanced accuracy and accuracy. A gain
is augmented minus real-only performance in the same unit. Tables use absolute
percentage points, not relative percent improvement.

A network beats a control on a dataset when the paired difference of gains
has positive mean, is positive in at least 3 of the 5 training sizes, and the
lower 95% bootstrap bound is above zero. There are 10,000 bootstrap resamples
of seeds within each training size. A network meets the main success criterion
only if it beats box center, interpolation and jitter and has positive macro-F1
gain and nonnegative balanced-accuracy gain. Neither network did so on any
dataset. This does not mean every individual control beats every network.

Against interpolation, 27/30 network × dataset × training-size blocks have
negative mean contrasts. For GAN/H3, the aggregate contrast is −0.10 pp with
95% interval [−0.31,+0.11], so superiority is not established; this is not a
formal equivalence test. Other network/interpolation contrasts are negative
with intervals below zero. Published intervals are descriptive, with no
multiple-comparison correction.

DDPM/H3 samples concentrate near the training mean (relative dispersion about
0.16), despite satisfying component bounds. The GAN and DDPM can break spin
symmetries obeyed by real data. In the earlier v2 configuration, the GAN barely
moved from initialization and behaved much like box center, while DDPM
memorized real samples. These failures motivated v3 and temper the interpretation
of positive gains over the real-only classifier.

Alternative classifiers do not reverse the overall reading: networks beat all
four controls in only 2/18 dataset/network/classifier combinations (GAN/H3
with logistic regression; DDPM/H1 with random forest). These are descriptive
exceptions without multiplicity correction. Copies often match or beat neural
gains for SVM/logistic regression, consistent with sensitivity to sample
multiplicity. In 3-NN neither network beats all controls on any dataset;
copies recover the real-only 1-NN predictions. These auxiliary results are
context, not a different primary endpoint.

## Data limitations and unresolved work

Seven ED-21 rows with |C|>1 were excluded on loading; source CSVs were not
modified. H1 loses one row, H3 six, H2 none. The selected tables use the grid
rerun after this exclusion. The configurations were not retuned.

Further independent diagonalization identified inconsistencies at degenerate
points (H2's Delta=−1 line and H3's ferromagnetic region). Some recorded vectors
cannot represent any state. An additional H1 point (Jz=−4,D=3.8) differed from
independent ED in a 900-point check. Removing the seven out-of-bound rows does
not fix these problems. Full verification of all 17,318 original rows and the
choice of convention within degenerate ground spaces remain unresolved.

Consequently these results establish behavior on the recorded datasets, not
that the source physical data have been fully validated. Dataset reconstruction
would require an explicit new dataset and a new experimental evaluation. No
corrected-dataset grid was run for this portfolio.

Other limits: finite-size calculations, inferred positional column conventions,
only three related Hamiltonians, sparse labeled training sets, one primary
classifier, and no v3 DMRG/ED-25 study. Neither phase discovery, quantum advantage,
general generator superiority nor experimental accessibility is established.

## Public demonstration versus research

The demo is independently generated L=4 ED with nine correlators and labels
assigned by sampled parameter regime. It uses an equal mixture of numerically
degenerate ground states. This is an explicit demonstration convention, not a
retroactive choice for the unresolved L=12 data. Its 40-step network budgets,
one seed and three training points per regime are execution checks. Demo
metrics cannot be substituted for the selected research aggregates.

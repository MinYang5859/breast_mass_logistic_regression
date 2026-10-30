# Breast Mass Classification: Logistic Regression and Decision Thresholds

## 1. From scores to predictions

Logistic regression converts a linear score into a probability using the
sigmoid function. A threshold then converts that probability into a class
label. For Case A, z = -1 + 0.8(2) = 0.6, so
p = 1 / (1 + exp(-0.6)) = 0.6457. Since this exceeds 0.50, the predicted
class is 1.

| Case | Intercept | Coefficient | x | Score z | Probability | Threshold | Class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.6 | 0.6457 | 0.50 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.6 | 0.6457 | 0.70 | 0 |
| C | -1.0 | -0.8 | 2.0 | -2.6 | 0.0691 | 0.50 | 0 |
| D | 1.0 | 0.0 | 2.0 | 1.0 | 0.7311 | 0.50 | 1 |

Cases A and B have identical probabilities but different labels because
their thresholds differ. Reversing the coefficient makes probability
decrease instead of increase as x increases. In Case D, the zero
coefficient removes the effect of x, giving a constant probability.
These coefficients are manually specified; trained coefficients are
learned by minimizing loss.

## 2. Data and experimental design

I used the Wisconsin Diagnostic Breast Cancer dataset loaded through
scikit-learn (scikit-learn developers, n.d.-a). Only mean radius and mean
texture were selected, in that order. The target was recoded as
1 = malignant and 0 = benign. There were 569 samples: 212 malignant and
357 benign. The selected features and target had no missing values.

Two stratified index splits used random_state=42. The first reserved
20% for test; the second reserved 25% of the remaining development rows
for validation. This produced 341 training, 114 validation, and 114 test
samples. Checks confirmed no overlapping or missing rows. StandardScaler
was fitted only on training features and reused for all partitions.
Test performance was not examined until the models and threshold were fixed.

## 3. Model learning and coefficients

The NumPy model used full-batch gradient descent, zero initialization,
a learning rate of 0.1, and 20,000 epochs, without regularization or class
weights. Mean binary cross-entropy decreased from 0.693147181 to
0.246833612. The loss curve flattened, and the final gradient norm was
approximately 3.24e-15.

Cross-entropy penalizes probabilities that disagree with observed labels.
Minimizing it is equivalent to maximizing the likelihood of those labels
under the model (scikit-learn developers, n.d.-b). Gradient descent adjusts
the coefficients to reduce this penalty.

| Standardized-unit coefficient | NumPy | scikit-learn |
|---|---:|---:|
| Intercept | -0.727172404 | -0.727172403 |
| mean radius | 3.656794172 | 3.656794174 |
| mean texture | 1.009871533 | 1.009871532 |

The unregularized scikit-learn reference used C=np.inf and solver="lbfgs".
The mean absolute difference between validation probabilities was about
1.19e-10. This close agreement supports the NumPy implementation.

Holding mean texture constant, a one-training-standard-deviation increase
in mean radius is associated with a 3.656794172 increase in malignant
log-odds. This is neither a fixed probability increase nor a causal effect.

## 4. Baseline and classification errors

The prior baseline assigned every sample a malignant probability of
0.372434018. At threshold 0.50, it predicted every sample as benign.
Its test accuracy was 0.631579, but recall and F1 were zero. Accuracy alone
therefore hides its failure to identify malignant samples. With no positive
predictions, precision has a zero denominator; zero_division=0 returns zero.

The NumPy test confusion matrix at the selected threshold was:

| Actual / Predicted | Benign (0) | Malignant (1) |
|---|---:|---:|
| Benign (0) | 69 (TN) | 3 (FP) |
| Malignant (1) | 14 (FN) | 28 (TP) |

The model correctly identified 28 malignant samples but missed 14.

## 5. Threshold selection and final results

The seven required thresholds were evaluated using NumPy validation F1.
Threshold 0.60 had the highest F1, 0.831169, and was fixed for both logistic
models. Ties would favor the smallest threshold. No model was retrained
using combined training and validation data.

On test data, increasing the threshold from 0.50 to 0.60 reduced FP from
4 to 3 but increased FN from 11 to 14. Precision increased from 0.885714
to 0.903226, while recall decreased from 0.738095 to 0.666667 and F1 fell
from 0.805195 to 0.767123. The validation improvement did not carry over
to test data. Both logistic models produced the same classification
results. Log loss remained 0.255657 because changing the threshold does
not change probabilities.

## 6. Scope and limitations

This is an educational experiment, not a clinical tool. False negatives
could create false reassurance and delay further assessment if predictions
were used clinically. Only two features and a limited sample were used.
Performance on other populations is unknown, calibration was not evaluated,
and there was no external validation. F1 was the assignment's selection
rule, not a clinical cost function; it does not establish that FP and FN
have equal consequences.

## Research process

An actual setup problem occurred when I tried to use the PowerShell
activation script in a CMD terminal. I checked the terminal details,
identified cmd.exe, and used .venv\Scripts\activate.bat instead. The
(.venv) prompt and sys.executable path confirmed the correct environment.

I also checked the NumPy implementation against an independently fitted
scikit-learn reference. The coefficient differences were approximately
1e-9, and the validation probability difference was approximately 1e-10.
These checks supported keeping the prescribed training settings.

## Sources and assistance

The experiment settings and four manual cases followed the course
assignment. ChatGPT assisted with step-by-step code development,
troubleshooting, and report drafting. The numerical results reported here
came from running the analysis script locally.

## References

scikit-learn developers. (n.d.-a). *load_breast_cancer*.
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

scikit-learn developers. (n.d.-b). *log_loss*.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
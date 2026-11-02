# Breast Mass Classification: Logistic Regression and Decision Thresholds

Can a logistic regression model built with NumPy reproduce a scikit-learn
reference, and how does the classification threshold affect its errors?

This project compares a manually implemented NumPy logistic regression,
an unregularized scikit-learn reference, and a majority-class baseline.
It uses two features: mean radius and mean texture.

**This project is for education only and must not be used for clinical decisions.**

## Main findings

- NumPy and scikit-learn validation probabilities differed by approximately
  1.19e-10 on average.
- Training loss decreased from 0.693147181 to 0.246833612.
- Validation F1 selected a threshold of 0.60.
- The selected threshold improved validation F1 but reduced test F1.
- On test data, increasing the threshold reduced false positives while
  increasing false negatives.

The following test results apply to both logistic regression models:

| Policy | Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 68 | 4 | 11 | 31 | 0.868421 | 0.885714 | 0.738095 | 0.805195 | 0.255657 |
| Selected | 0.60 | 69 | 3 | 14 | 28 | 0.850877 | 0.903226 | 0.666667 | 0.767123 | 0.255657 |

The baseline test accuracy was 0.631579, but its malignant recall and F1
were both zero.

Read the [full analysis memo](memo.md) for calculations, coefficient
interpretations, the confusion matrix, limitations, and the research process.

## Dataset and class definitions

The analysis uses the Wisconsin Diagnostic Breast Cancer dataset supplied
with scikit-learn.

- Dataset loader: `sklearn.datasets.load_breast_cancer`
- Selected features, in order: `mean radius`, `mean texture`
- Samples: 569
- Benign samples: 357
- Malignant samples: 212
- Positive class: **1 = malignant**
- Negative class: **0 = benign**
- No missing values were found in the selected features or target.

The script explicitly recodes the original target labels.
No separate dataset download is needed, and no dataset copy is included
in this repository.

Dataset documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Experiment design

Two stratified splits use `random_state=42`:

1. Reserve 20% of all row indices for test.
2. Reserve 25% of the remaining development indices for validation.

The resulting partitions contain:

| Partition | Total | Benign (0) | Malignant (1) |
|---|---:|---:|---:|
| Training | 341 | 214 | 127 |
| Validation | 114 | 71 | 43 |
| Test | 114 | 72 | 42 |

The script checks that partitions do not overlap and cover every row.
One StandardScaler is fitted only on training features and then reused.

NumPy training settings:

- Full-batch gradient descent
- Mean binary cross-entropy
- Intercept and feature coefficients initialized to zero
- Learning rate: 0.1
- Epochs: 20,000
- No regularization or class weights
- Stable sigmoid and loss calculations using `np.logaddexp`

Reference settings:

```python
LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9,
)
```

The baseline uses `DummyClassifier(strategy="prior")`.

Threshold candidates are 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, and 0.80.
Selection uses NumPy validation F1 only; ties favor the smallest threshold.
Both logistic models use the same selected threshold.

Models are not retrained after threshold selection.
Test results are used only for final evaluation.

## Files

| File | Purpose |
|---|---|
| `logistic_regression_analysis.py` | Complete analysis from data loading to exports |
| `requirements.txt` | Pinned package versions |
| `memo.md` | Written analysis and calculations |
| `reports/threshold_results.csv` | Seven validation threshold candidates |
| `reports/assignment_results.csv` | Ten final evaluation configurations |
| `reports/training_loss.png` | NumPy training loss from epoch 0 to 20,000 |
| `.gitignore` | Excludes environments, caches, credentials, and private forms |

## Software requirements

You need Python, Git, internet access for installation and cloning,
and a terminal. VS Code is optional.

The analysis has been run locally on Windows using:

| Software | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |

Other installed package versions are recorded in `requirements.txt`.

macOS instructions are provided below, but macOS execution has not yet
been verified. Independent peer validation is pending.

### Install Python and Git

Python downloads:
https://www.python.org/downloads/

Git installation instructions:
https://git-scm.com/downloads

Install Python 3.13 to match the tested Python minor version.
On Windows, enable the option to add Python to PATH if offered.
After installation, close and reopen your terminal.

## Windows: PowerShell instructions

### 1. Open PowerShell and check installation

Open the Start menu, search for PowerShell, and open it.
Enter each command separately:

```powershell
py --version
git --version
```

If `py` is unavailable but `python --version` works, replace `py` with
`python` in the environment creation command below.

### 2. Clone the repository

Choose a parent folder where you want to keep the project, then run:

```powershell
cd "$HOME\Documents"
git clone https://github.com/MinYang5859/breast_mass_logistic_regression.git breast_mass_logistic_regression
cd breast_mass_logistic_regression
```

The final `cd` enters the project root, which contains
`logistic_regression_analysis.py` and `requirements.txt`.

Repository: [breast_mass_logistic_regression](https://github.com/MinYang5859/breast_mass_logistic_regression)

### 3. Create and activate an environment

Run from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell reports that running scripts is disabled, activation is
optional. You can continue to Step 4 and use the environment's Python
directly without changing the execution policy.

Alternatively, to activate the environment in the current PowerShell
session, run these commands one at a time:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If a confirmation prompt appears, enter `Y` and press Enter.
The policy change applies only to this PowerShell session and expires
when it closes. It does not permanently change your execution policy.

If an organizational policy still blocks activation, use the direct
Python commands in Step 4.

These commands are for PowerShell only. For Command Prompt, use the
Windows CMD alternative below.

Reference:
https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_execution_policies

### 4. Install packages and run the analysis

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe logistic_regression_analysis.py
```

These commands work even if activation was blocked.

To check the interpreter:

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

The path should end with `.venv\Scripts\python.exe`.

### Windows CMD alternative

If your terminal is Command Prompt rather than PowerShell, activate with:

```bat
.\.venv\Scripts\activate.bat
```

Then run:

```bat
python -m pip install -r requirements.txt
python logistic_regression_analysis.py
```

Do not use `Activate.ps1` in CMD.

## macOS: Terminal instructions

### 1. Open Terminal and check installation

Open Spotlight with Command + Space, type Terminal, and open it.

```bash
python3.13 --version
git --version
```

If `python3.13` is unavailable, check `python3 --version`.
If that reports Python 3.13, use `python3` in the environment creation
command below.

### 2. Clone the repository

Run from your chosen parent folder:

```bash
cd "$HOME/Documents"
git clone https://github.com/MinYang5859/breast_mass_logistic_regression.git breast_mass_logistic_regression
cd breast_mass_logistic_regression
```



### 3. Create and activate an environment

Run from the project root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

### 4. Install packages and run

Run from the project root:

```bash
python -m pip install -r requirements.txt
python logistic_regression_analysis.py
```

To check the interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

The path should end with `.venv/bin/python`.

## Expected output and verification

Measured analysis runtime on the author's Windows computer:
approximately 3.64 seconds, including Python startup,
model training, evaluation, and output generation.
Package installation is excluded. Runtime varies by computer.

The terminal prints:

1. Four manual score, probability, and class calculations.
2. Feature names, missing-value checks, and class counts.
3. Partition counts and scaling checks.
4. NumPy training loss and learned coefficients.
5. Reference coefficient and validation probability comparisons.
6. Seven threshold candidates and the selected threshold.
7. Five validation and five test evaluation configurations.
8. Output file locations.

Expected checks include:

- Initial training loss: approximately 0.693147181
- Final training loss: approximately 0.246833612
- Loss values recorded: 20,001
- Selected threshold: 0.60
- NumPy coefficients: approximately -0.727172404, 3.656794172,
  and 1.009871533

Small floating-point differences may occur across computers.
For reproduction checks, an absolute difference up to 0.000001 in
coefficients or reported metrics is acceptable. Partition counts,
confusion counts, threshold selection, and CSV structure should match.

A larger difference should be investigated rather than silently accepted.

### Generated files

Outputs are saved beside the script in `reports/`, regardless of the
terminal's working directory. Re-running overwrites the generated outputs.

| Output | Expected structure |
|---|---|
| `threshold_results.csv` | Header plus 7 data rows |
| `assignment_results.csv` | Header plus 10 data rows |
| `training_loss.png` | Labeled loss-versus-epoch plot |

CSV metrics use decimals, not percentages. Confusion counts are integers.
The classification rule is `probability >= threshold`.

Log loss uses probabilities. Precision, recall, and F1 use
`zero_division=0`.

Open `reports` in File Explorer on Windows or Finder on macOS.
CSV files can be opened in a text editor or spreadsheet application.
The PNG opens in an image viewer.

You can also open the folder from the project root:

Windows PowerShell:

```powershell
explorer.exe .\reports
```

macOS Terminal:

```bash
open reports
```

## Troubleshooting

### Python command is not recognized

Close and reopen the terminal after installing Python.
On Windows, try `py --version` and `python --version`.
If neither works, check the Python installation and PATH configuration.

On macOS, try `python3.13 --version` and `python3 --version`.
Use the Python 3.13 command available on your system.

### ModuleNotFoundError or missing dependencies

From the project root, reinstall into the project environment.

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

### Script or requirements file cannot be found

Check your current folder.

Windows PowerShell:

```powershell
Get-Location
Get-ChildItem
```

macOS:

```bash
pwd
ls
```

Enter the cloned project folder before running the installation and
analysis commands. You should see `logistic_regression_analysis.py`.

### PermissionError when exporting a CSV

Close the generated CSV files in Excel or other applications, then run
the script again.

### Pinned package installation fails

Check that Python is version 3.13 and read the first installation error.
Record the operating system, Python version, and error message.
Do not claim successful replication if installation remains blocked.

## Limitations

Only two features and one fixed split were evaluated. The sample is limited,
performance on other populations is unknown, probability calibration was
not evaluated, and external validation was not performed.

F1 selection is an educational rule, not a clinical assessment of the costs
of false positives and false negatives.

## Independent replication

Status: pending.

A reviewer must obtain a fresh repository copy and reproduce the analysis
using this README without live assistance from the author.
The tested operating system, Python version, branch, and final successful
commit must be recorded on the course verification form.

Completed forms, signatures, and reviewer personal information are kept
outside this repository and submitted privately through Canvas.

## Scope and acknowledgments

This repository contains the required experiment only, with no optional
model extensions.

ChatGPT assisted with code development, troubleshooting, and documentation.
Results were generated by running the analysis locally.
See [memo.md](memo.md) for references and the research process.
# PhishGuard --- Antigravity Implementation Workflow

**Project title:** PhishGuard --- Explainable Machine Learning Based
Phishing URL Detection System\
**Purpose:** Build a reproducible, properly evaluated machine-learning
project and a usable web application that accepts a URL and predicts
whether it is phishing or legitimate.

> **Important:** This file is the project specification. Follow it in
> stages. Do not generate the entire project in one response, do not
> invent results, and do not silently skip a requirement. Antigravity
> must inspect the actual dataset and current project files before
> making implementation decisions.

------------------------------------------------------------------------

## 1. Instructions for Antigravity

Act as a careful ML engineer, Python developer, reviewer, and tutor.
Work in the user's existing project workspace.

### Required working style

1.  **Work incrementally.** Complete only the current phase, then stop
    and report what was created, what was run, actual outputs, issues,
    and the exact next action.
2.  **Do not attempt the entire project in one response.** Use the
    staged plan in Section 4. Wait for the user to ask you to continue
    before starting the next phase.
3.  **Inspect before changing.** List and inspect existing files and the
    dataset first. Do not overwrite useful work without explaining why
    and obtaining approval where appropriate.
4.  **Use actual evidence.** Run the relevant code/tests and report real
    outputs. Never invent accuracy, metrics, plots, file contents,
    dataset facts, citations, licences, or deployment status.
5.  **Explain decisions.** For every major decision, explain what was
    done, why, how it was implemented, what changed, and how it was
    validated.
6.  **Keep the implementation understandable.** Use clear function
    names, concise comments, meaningful Markdown, and explanations
    suitable for a student viva.
7.  **Respect scope.** The only permitted modelling algorithms are
    Logistic Regression and Decision Tree Classifier. Do not add Random
    Forest, XGBoost, SVM, neural networks, or other models unless the
    user explicitly changes this requirement.
8.  **Do not fabricate a raw URL or label.** If the provided dataset
    cannot support training a URL-only predictor, stop at the
    feasibility gate, explain the issue, and present evidence-backed
    choices. Never create synthetic URLs or labels and pretend they are
    real data.
9.  **Do not use external website crawling or live website fetching in
    the web app.** The application should extract features from the
    submitted URL string only, unless the user explicitly approves a
    different, documented design.
10. **Never claim a model or app is deployed unless it was actually
    deployed and verified.**

### Initial workspace assumption

The project workspace is the folder in which the user opens Antigravity.
The only guaranteed input artifact for a fresh start is the provided CSV
dataset. Detect the actual path and filename instead of assuming a fixed
filename or drive path. The CSV may appear as
`PhiUSIIL_Phishing_30000(1).csv`; use the actual file found in the
workspace and keep its name configurable.

Do not mention dataset sampling or reduction in project documentation.
Describe and analyze the working CSV that is actually supplied.

------------------------------------------------------------------------

## 2. Non-negotiable assignment requirements

The Machine Learning Innovative Assignment guidelines emphasize a
complete, justified, reproducible and usable ML solution---not only a
high accuracy score.

The project must cover:

-   Problem definition and use case
-   Data source, ownership/licence and dataset understanding
-   Data preparation and duplicate/error checks
-   Preprocessing: missing values, outliers, categorical values,
    encoding/scaling where needed
-   Class-distribution analysis and justified balancing decision
-   Analytical EDA with distributions, associations, correlations,
    anomalies and decisions based on findings
-   Feature engineering and selection with reasons for
    retaining/excluding features
-   Baseline modelling
-   At least two meaningful modelling approaches where appropriate
-   Hyperparameter tuning and validation
-   Suitable evaluation metrics and error analysis
-   Application, testing, use case, impact, limitations and future
    improvements
-   Reproducibility and an auditable record of data sources, libraries,
    and major decisions

For every major decision, document: 1. What was done? 2. Why was it
done? 3. How was it implemented? 4. What changed? 5. How was it
validated?

Never claim improvement solely because training accuracy is higher.
Never report fabricated results. Accuracy alone is insufficient.

------------------------------------------------------------------------

## 3. Target-label convention --- MUST remain consistent

The dataset's documented convention for this project is:

-   Original `label = 0` means **Phishing**
-   Original `label = 1` means **Legitimate**

Create the project target:

``` python
df["is_phishing"] = (df["label"] == 0).astype("int8")
```

Therefore:

-   `is_phishing = 1` means **Phishing**
-   `is_phishing = 0` means **Legitimate**
-   The positive class is **Phishing**

Validate the original labels before transforming them. Do not silently
reverse this mapping. Keep this convention consistent in plots, metrics,
confusion matrices, feature explanations, model output, Streamlit UI,
and documentation.

------------------------------------------------------------------------

## 4. Fourteen-step project roadmap

These are 14 implementation steps, organized to cover the assignment
pipeline and a tested web application. They do not replace any mandatory
assignment stage.

### Step 1 --- Workspace audit and project skeleton

-   Inspect the workspace, CSV, existing files, Python version, and
    installed packages.
-   Confirm the dataset is readable and determine its actual filename
    and location.
-   Create a clean, understandable folder structure only after reporting
    what already exists.
-   Create `requirements.txt`, `.gitignore`, `README.md`, and a
    configuration module.
-   Do not start modelling yet.

### Step 2 --- Problem definition and use case

-   Define the phishing URL problem, intended users, input, output,
    constraints, risks, and expected impact.
-   Specify that the app gives a prediction, not a guarantee of safety.
-   Record the target mapping and the cost of false negatives.

### Step 3 --- Data acquisition and understanding

-   Inspect source information available with the supplied dataset.
-   Verify source URL, collection/ownership information, licence or
    usage conditions, and access date where possible.
-   If source/licence evidence is unavailable, record it as unverified;
    do not invent it.
-   Report rows, columns, data types, target, and feature definitions
    when available.

### Step 4 --- Data preparation and quality audit

-   Inspect missing values, duplicates, invalid values, target validity,
    and repeated/conflicting records.
-   Record row/column counts before and after every change.
-   Preserve the original CSV unchanged.
-   Investigate repeated URLs/domains and conflicting labels if raw
    URLs/domains are present.
-   Never automatically remove records without documenting and
    validating the reason.

### Step 5 --- Preprocessing and leakage audit

-   Inspect numerical, categorical, binary, identifier-like, and target
    columns.
-   Investigate potential leakage, including `FILENAME` if present.
-   Examine `IsHTTPS` and similar features for provenance and possible
    dataset artifacts.
-   Flag outliers using appropriate methods; do not automatically remove
    valid extreme observations.
-   Decide whether encoding/scaling is needed based on the selected
    algorithm.
-   All learned transformations must be fitted only on training data or
    inside cross-validation pipelines.

### Step 6 --- Class distribution and balancing

-   Calculate counts and percentages for legitimate and phishing
    samples.
-   Explain whether imbalance is meaningful for this dataset.
-   Do not apply SMOTE/over- or undersampling automatically.
-   Start with the original distribution. Consider class weighting only
    if training-fold validation supports it.
-   Any resampling, if later justified, must occur inside training folds
    only---not on validation/test data.

### Step 7 --- Exploratory Data Analysis (EDA)

-   Target distribution.
-   Descriptive statistics and selected numerical distributions.
-   Binary-feature prevalence.
-   Feature relationships by target class.
-   Correlation/association analysis where meaningful.
-   Categorical value frequencies where useful.
-   Outlier and anomaly investigation.
-   Potential leakage and feature-reproducibility analysis.
-   Every plot/table must answer a question and lead to a documented
    observation or decision. Avoid decorative graphs.
-   Save plots and tables under `results/`.

### Step 8 --- Feature engineering and selection

-   Create and document two feature groups:
    -   **FS-REF (research/reference):** dataset features useful for
        analysis but not necessarily available from a raw URL.
    -   **FS-URL (deployable):** features that can be computed
        deterministically from the user-submitted URL string.
-   Build a feature dictionary containing feature name, definition,
    source, data type, target leakage risk, URL-only availability, and
    decision.
-   Exclude target, identifiers, label proxies, and confirmed leakage
    features from predictors.
-   Select features using domain reasoning and training-only evidence.
    Do not use the final test set to choose features.
-   Do not assume a page-derived feature can be extracted from a URL
    string.
-   **Critical feasibility gate:** establish whether the supplied CSV
    contains original raw URLs or sufficient verified information to
    train a model using the exact FS-URL extractor. If it does not, do
    not pretend an FS-URL model can be trained from it. Stop, show the
    evidence, and explain the legitimate choices (for example, obtain an
    additional labelled URL dataset or revise the scope with the user's
    approval). Never generate fake raw URLs or fake labels.

### Step 9 --- Reproducible data split and baseline model

-   Only after Step 8's feasibility gate passes, prepare the modelling
    dataset.
-   Use an 80/20 stratified train/test split with `random_state=42`,
    unless actual evidence requires a documented alternative.
-   Lock the test set until the final evaluation.
-   Use 5-fold stratified cross-validation on training data for model
    selection.
-   Establish a simple, meaningful baseline before claiming
    improvements.
-   Keep preprocessing in an sklearn Pipeline where applicable.

### Step 10 --- Logistic Regression model

-   Use Logistic Regression as one of the two permitted algorithms.
-   Justify suitability, assumptions, and preprocessing.
-   Use scaling/encoding inside the pipeline when needed.
-   Evaluate using training-only cross-validation and record actual
    results.
-   Do not claim Logistic Regression is better until a fair comparison
    exists.

### Step 11 --- Decision Tree model

-   Use `DecisionTreeClassifier` as the second permitted algorithm.
-   Explain interpretability, non-linearity, and overfitting risks.
-   Tune a practical grid using training-only cross-validation. An
    initial grid may include:
    -   `criterion`: `gini`, `entropy`
    -   `max_depth`: 3, 5, 8, 12
    -   `min_samples_leaf`: 1, 5, 10
    -   `min_samples_split`: 2, 10
-   Adjust the grid only for a documented reason and report the actual
    search.
-   Do not use test-set scores to tune parameters.

### Step 12 --- Model comparison and final evaluation

-   Compare Logistic Regression and Decision Tree fairly using the same
    training folds and relevant metrics.
-   Select the final model based on the project's error costs and
    validation evidence, not accuracy alone.
-   After choices are locked, evaluate once on the held-out test set.
-   Report accuracy, precision, recall, F1-score, confusion matrix,
    ROC-AUC and PR-AUC where technically appropriate.
-   Treat phishing (`is_phishing=1`) as the positive class.
-   Discuss false negatives (phishing predicted as legitimate) and false
    positives.
-   Consider threshold implications where applicable.
-   A Decision Tree's `predict_proba` output must be called **model
    probability**, not guaranteed real-world confidence; do not claim
    calibration unless tested.
-   Save actual metrics, plots, model details, and error examples.

### Step 13 --- Save artifacts and build the Streamlit application

-   Save the final fitted preprocessing/model pipeline to
    `models/final_pipeline.joblib`.
-   Save a feature schema, feature order, target mapping, model
    metadata, and reproducibility settings.
-   Implement a deterministic URL-only feature extractor in `src/`.
-   The extractor and training pipeline must produce the same feature
    names, order, types, and semantics.
-   The Streamlit application must:
    1.  Accept a URL through a text input.
    2.  Validate and normalize the input safely.
    3.  Extract only the features defined in FS-URL.
    4.  Construct the feature vector in the saved feature order.
    5.  Load the saved fitted pipeline and predict.
    6.  Display **Phishing** or **Legitimate**, plus a sensible
        explanation/warning.
    7.  Handle invalid input and missing artifacts gracefully.
-   Do not train the model inside Streamlit.
-   Do not fetch/crawl the submitted website or invent missing feature
    values.
-   If showing `predict_proba`, label it "Model Probability" and explain
    that it may not be calibrated.
-   Do not expose stack traces, file paths, secrets, or internal errors
    to the user.

### Step 14 --- Testing, documentation and final verification

-   Unit-test the URL extractor, target mapping, feature order, input
    validation, artifact loading, and predictions.
-   Test representative valid URLs and malformed inputs.
-   Verify that the same URL produces the same extracted feature vector.
-   Verify that the app's feature schema exactly matches the training
    schema.
-   Run tests and report actual pass/fail results.
-   Update `README.md` with setup, data placement, environment creation,
    training, evaluation, running the app, limitations, and
    reproducibility.
-   Document dataset source/licence accurately and list dependencies.
-   Test the app locally using the actual saved model.
-   Do not claim deployment unless a real deployment was completed and
    verified.

------------------------------------------------------------------------

## 5. Required project structure

Adapt this structure to the actual workspace without deleting useful
files:

``` text
PhishGuard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── <provided CSV>
│   └── processed/
├── models/
├── notebooks/
│   ├── 01_problem_data_audit.ipynb
│   ├── 02_eda.ipynb
│   └── 03_feature_engineering.ipynb
├── results/
│   ├── figures/
│   ├── metrics/
│   └── tables/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── fixtures/
│   ├── test_data_validation.py
│   ├── test_feature_extraction.py
│   ├── test_pipeline_schema.py
│   └── test_prediction.py
├── .gitignore
├── requirements.txt
├── README.md
└── WORKFLOW.md
```

Create files only when their phase is reached. Do not generate empty
placeholder files just to fill the tree. Add or remove modules only with
an explained reason.

------------------------------------------------------------------------

## 6. Data and feature rules

### Dataset handling

-   Detect and use the actual provided CSV file.
-   Preserve raw data; save cleaned/derived datasets separately.
-   Record before/after row and column counts.
-   Check missing values, duplicates, target distribution, data types,
    suspicious identifiers, and invalid values.
-   Do not write dataset facts into documentation until they have been
    measured from the actual file.
-   Do not disclose or describe any sampling/reduction history in
    project documentation.

### Feature groups

**FS-REF:** may include page-level features for dataset analysis and
reference, but they are not automatically eligible for the final
URL-input model.

**FS-URL:** only deterministic features computed from the submitted URL
string. Examples, subject to implementation and dataset compatibility,
include URL length, hostname/domain length, counts of digits and
selected special characters, whether an IP address is used as hostname,
presence of a non-default port, number of subdomains, and HTTPS scheme.
Each feature must have a clear definition and robust tests.

Do not invent unavailable features such as page title, HTML line count,
external references, favicon status, or page responsiveness from a URL
string. Do not use network fetching to make these features appear
available.

### Dataset feasibility rule

The provided dataset may consist largely or entirely of engineered
features. First inspect whether raw URLs are available and whether the
proposed extractor's feature definitions align with the dataset. If no
valid labelled URL rows are available to train the same FS-URL features,
the app's requested prediction is not yet supported by that file alone.
Report this limitation clearly and ask for approval before adding
another dataset or changing scope. Do not quietly train a page-feature
model and connect it to a URL-only app.

------------------------------------------------------------------------

## 7. Modelling and validation rules

-   Permitted models: Logistic Regression and Decision Tree Classifier
    only.
-   Use `random_state=42` wherever relevant.
-   Default split: stratified 80% training / 20% final test.
-   Use 5-fold stratified CV on training data for model
    comparison/tuning when the class counts allow it.
-   Keep the test set locked until feature decisions and tuning are
    complete.
-   If duplicate URLs/domains exist, investigate split leakage and
    consider grouped robustness testing where feasible.
-   Use sklearn Pipelines and `ColumnTransformer` as appropriate so
    transformations are learned only from training folds.
-   Start without resampling. Any later balancing must be justified by
    CV evidence and applied only inside training folds.
-   Do not select features or thresholds based on the final test set.
-   Report metrics for the positive phishing class and inspect the
    confusion matrix.
-   Save random seeds, configuration, feature lists, cross-validation
    results, test metrics, and chosen hyperparameters.
-   Never promise a particular accuracy before experiments are run.
    "Maintaining proper accuracy" means optimizing honestly while
    monitoring phishing recall, precision, F1, false negatives,
    generalization, and deployment validity---not manipulating results.

------------------------------------------------------------------------

## 8. Web application acceptance criteria

The application is complete only when all of the following work:

-   A user can enter a URL in a Streamlit text box.
-   Input validation handles empty/malformed input.
-   The feature extractor returns exactly the features expected by the
    fitted pipeline.
-   The saved model predicts without retraining.
-   The app displays a clear phishing/legitimate result.
-   The app communicates uncertainty and does not promise safety.
-   No page scraping, network fetching, or made-up feature values are
    used.
-   Missing model artifacts produce a helpful error message.
-   Automated tests cover URL parsing, schema consistency, and
    prediction flow.
-   The application runs locally using documented commands.

------------------------------------------------------------------------

## 9. Reproducibility and deliverables

By the end, keep these deliverables where applicable:

-   `WORKFLOW.md` --- this specification
-   `README.md` --- setup and usage instructions
-   Dataset source/licence notes
-   Data audit report and cleaned dataset, if cleaning is justified
-   EDA figures and tables
-   Feature dictionary with FS-REF/FS-URL decisions
-   Reproducible train/validation/test protocol
-   Cross-validation and tuning results
-   Final held-out test metrics and confusion matrix
-   Error-analysis examples
-   Saved model pipeline and feature schema
-   Streamlit application
-   Automated tests and their actual results
-   Dependency list and source/reference log

Do not commit virtual environments, secrets, credentials, or unnecessary
large artifacts to version control. Do not publish the dataset or model
unless the applicable licence permits it.

------------------------------------------------------------------------

## 10. Antigravity phase-by-phase interaction protocol

At the start of every phase:

1.  State the current step and its acceptance criteria.
2.  Inspect relevant existing files and actual outputs.
3.  Implement only the current step.
4.  Run checks or code relevant to that step.
5.  Summarize:
    -   files created/modified,
    -   commands run,
    -   actual outputs and observations,
    -   issues/limitations,
    -   decisions and reasons,
    -   acceptance criteria completed/not completed.
6.  Stop and ask the user to review or explicitly continue.

Do not continue to the next phase automatically. If a critical issue is
found---especially lack of raw URLs or a mismatch between dataset
features and URL-extractor features---stop and explain it before
proceeding.

### First prompt / first response

Begin with **Step 1 only**. Inspect the current workspace and the
provided CSV. Do not write the entire project yet. Report: - the actual
workspace path, - files and folders currently present, - CSV path,
dimensions, columns, data types, target values, and missingness, -
whether a raw URL column is present, - whether the provided data can
legitimately support training the same URL-only features the app will
extract, - any dataset-source/licence information that can be verified
from supplied materials.

Then propose the minimal Step 1 skeleton and wait for the user's
approval before proceeding to Step 2. Do not fabricate or infer
unobserved dataset properties.

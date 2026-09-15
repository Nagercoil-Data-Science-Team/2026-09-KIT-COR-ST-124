import os
import glob
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks

# ============================================================
# 0. OUTPUT FOLDERS (ALL CSV OUTPUTS / ALL NPY OUTPUTS GROUPED)
# ============================================================

CSV_OUTPUT_DIR = "outputs_csv"
NPY_OUTPUT_DIR = "outputs_npy"

os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)
os.makedirs(NPY_OUTPUT_DIR, exist_ok=True)

print("CSV output folder :", CSV_OUTPUT_DIR)
print("NPY output folder :", NPY_OUTPUT_DIR)

# ============================================================
# 1. DATA PATH
# ============================================================

BASE_PATH = "data/raw/phone"
ACCEL_PATH = os.path.join(BASE_PATH, "accel")
GYRO_PATH = os.path.join(BASE_PATH, "gyro")

# ============================================================
# 2. DISPLAY DATASET STRUCTURE
# ============================================================

print("=" * 70)
print("WISDM DATASET")
print("=" * 70)
print("Base path :", BASE_PATH)
print("Accel path:", ACCEL_PATH)
print("Gyro path :", GYRO_PATH)

# ============================================================
# 3. FIND TXT AND ARFF FILES
# ============================================================

accel_txt_files = sorted(glob.glob(os.path.join(ACCEL_PATH, "*.txt")))
accel_arff_files = sorted(glob.glob(os.path.join(ACCEL_PATH, "*.arff")))
gyro_txt_files = sorted(glob.glob(os.path.join(GYRO_PATH, "*.txt")))
gyro_arff_files = sorted(glob.glob(os.path.join(GYRO_PATH, "*.arff")))

print("\n" + "=" * 70)
print("FILE COUNTS")
print("=" * 70)
print("Accelerometer TXT :", len(accel_txt_files))
print("Accelerometer ARFF:", len(accel_arff_files))
print("Gyroscope TXT     :", len(gyro_txt_files))
print("Gyroscope ARFF    :", len(gyro_arff_files))

# ============================================================
# 4. FUNCTION TO LOAD WISDM RAW TXT FILE
# ============================================================

def load_txt_file(file_path, sensor_type):
    try:
        df = pd.read_csv(file_path, header=None, sep=",", engine="python")
    except Exception as e:
        print("Error reading:", file_path)
        print(e)
        return pd.DataFrame()

    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    if df.shape[1] >= 6:
        df.iloc[:, 5] = df.iloc[:, 5].astype(str).str.replace(";", "", regex=False).str.strip()

    df = df.iloc[:, :6]
    df.columns = ["subject_id", "activity", "timestamp", "x", "y", "z"]

    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["z"] = pd.to_numeric(df["z"], errors="coerce")

    df = df.dropna(subset=["subject_id", "activity", "timestamp", "x", "y", "z"])
    df["source_file"] = os.path.basename(file_path)

    if sensor_type == "accelerometer":
        df = df.rename(columns={"x": "accel_x", "y": "accel_y", "z": "accel_z"})

    if sensor_type == "gyroscope":
        df = df.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})

    return df

# ============================================================
# 5. LOAD ALL ACCELEROMETER TXT FILES
# ============================================================

accel_list = []

print("\n" + "=" * 70)
print("LOADING ACCELEROMETER TXT FILES")
print("=" * 70)

for file in accel_txt_files:
    df = load_txt_file(file, "accelerometer")
    if not df.empty:
        accel_list.append(df)
        print("Loaded:", os.path.basename(file), "| Shape:", df.shape)

if len(accel_list) > 0:
    accel_data = pd.concat(accel_list, ignore_index=True)
else:
    accel_data = pd.DataFrame()

# ============================================================
# 6. LOAD ALL GYROSCOPE TXT FILES
# ============================================================

gyro_list = []

print("\n" + "=" * 70)
print("LOADING GYROSCOPE TXT FILES")
print("=" * 70)

for file in gyro_txt_files:
    df = load_txt_file(file, "gyroscope")
    if not df.empty:
        gyro_list.append(df)
        print("Loaded:", os.path.basename(file), "| Shape:", df.shape)

if len(gyro_list) > 0:
    gyro_data = pd.concat(gyro_list, ignore_index=True)
else:
    gyro_data = pd.DataFrame()

# ============================================================
# 7. DISPLAY ACCELEROMETER DATA
# ============================================================

print("\n" + "=" * 70)
print("ACCELEROMETER DATA")
print("=" * 70)
print("Shape:", accel_data.shape)
print("\nColumns:")
print(accel_data.columns.tolist())
print("\nFirst 10 rows:")
print(accel_data.head(10))

# ============================================================
# 8. DISPLAY GYROSCOPE DATA
# ============================================================

print("\n" + "=" * 70)
print("GYROSCOPE DATA")
print("=" * 70)
print("Shape:", gyro_data.shape)
print("\nColumns:")
print(gyro_data.columns.tolist())
print("\nFirst 10 rows:")
print(gyro_data.head(10))

# ============================================================
# 9. SORT SENSOR DATA
# ============================================================

print("\n" + "=" * 70)
print("SORTING SENSOR DATA")
print("=" * 70)

accel_data = accel_data.sort_values(by=["subject_id", "activity", "timestamp"]).reset_index(drop=True)
gyro_data = gyro_data.sort_values(by=["subject_id", "activity", "timestamp"]).reset_index(drop=True)

print("Accelerometer sorted.")
print("Gyroscope sorted.")

# ============================================================
# 10. SUBJECT AND ACTIVITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SUBJECT AND ACTIVITY CHECK")
print("=" * 70)

print("Accelerometer subjects:", accel_data["subject_id"].nunique())
print("Gyroscope subjects:", gyro_data["subject_id"].nunique())
print("Accelerometer activities:", accel_data["activity"].nunique())
print("Gyroscope activities:", gyro_data["activity"].nunique())

# ============================================================
# 11. MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE CHECK")
print("=" * 70)

print("Accelerometer missing values:", accel_data.isnull().sum().to_dict())
print("Gyroscope missing values:", gyro_data.isnull().sum().to_dict())

# ============================================================
# 12. DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

accel_duplicates = accel_data.duplicated(subset=["subject_id", "activity", "timestamp"]).sum()
gyro_duplicates = gyro_data.duplicated(subset=["subject_id", "activity", "timestamp"]).sum()

print("Accelerometer duplicate timestamps:", accel_duplicates)
print("Gyroscope duplicate timestamps:", gyro_duplicates)

accel_data = accel_data.drop_duplicates(subset=["subject_id", "activity", "timestamp"]).reset_index(drop=True)
gyro_data = gyro_data.drop_duplicates(subset=["subject_id", "activity", "timestamp"]).reset_index(drop=True)

# ============================================================
# 13. PREPARE ACCELEROMETER DATA
# ============================================================

accel_merge = accel_data[["subject_id", "activity", "timestamp", "accel_x", "accel_y", "accel_z"]].copy()

# ============================================================
# 14. PREPARE GYROSCOPE DATA
# ============================================================

gyro_merge = gyro_data[["subject_id", "activity", "timestamp", "gyro_x", "gyro_y", "gyro_z"]].copy()

# ============================================================
# 15. SYNCHRONIZE ACCELEROMETER + GYROSCOPE
# ============================================================

print("\n" + "=" * 70)
print("SYNCHRONIZING ACCELEROMETER AND GYROSCOPE")
print("=" * 70)

print("Merge keys: subject_id + activity + timestamp")

sensor_data = pd.merge(accel_merge, gyro_merge, on=["subject_id", "activity", "timestamp"], how="inner")

# ============================================================
# 16. SORT SYNCHRONIZED DATA
# ============================================================

sensor_data = sensor_data.sort_values(by=["subject_id", "activity", "timestamp"]).reset_index(drop=True)

# ============================================================
# 17. ACTIVITY MAPPING
# ============================================================

activity_map = {"A": "Walking", "B": "Jogging", "C": "Stairs", "D": "Sitting", "E": "Standing", "F": "Typing", "G": "Brushing Teeth", "H": "Eating Soup", "I": "Eating Chips", "J": "Eating Pasta", "K": "Drinking", "L": "Eating Sandwich", "M": "Kicking", "O": "Playing Catch", "P": "Dribbling", "Q": "Writing", "R": "Clapping", "S": "Folding Clothes"}

sensor_data["activity_name"] = sensor_data["activity"].map(activity_map)

# ============================================================
# 18. DISPLAY ALL ACTIVITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ALL ACTIVITY DISTRIBUTION")
print("=" * 70)

print(sensor_data["activity_name"].value_counts())

# ============================================================
# 19. SELECT ONLY 5 ACTIVITIES
# ============================================================

selected_activities = ["Drinking", "Jogging", "Walking", "Kicking", "Dribbling"]

sensor_data = sensor_data[sensor_data["activity_name"].isin(selected_activities)].copy()

sensor_data = sensor_data.reset_index(drop=True)

# ============================================================
# 20. VERIFY 5-CLASS DATASET
# ============================================================

print("\n" + "=" * 70)
print("SELECTED 5-CLASS DATASET")
print("=" * 70)

print("Selected activities:", selected_activities)
print("Number of classes:", sensor_data["activity_name"].nunique())
print("Number of subjects:", sensor_data["subject_id"].nunique())
print("Total synchronized records:", len(sensor_data))

print("\n5-Class Distribution:")
print(sensor_data["activity_name"].value_counts())

# ============================================================
# 21. CHECK ACTIVITY CODES
# ============================================================

print("\n" + "=" * 70)
print("SELECTED ACTIVITY CODES")
print("=" * 70)

print(sensor_data[["activity", "activity_name"]].drop_duplicates().sort_values("activity"))

# ============================================================
# 22. CHECK SENSOR RANGES
# ============================================================

print("\n" + "=" * 70)
print("SENSOR RANGE CHECK")
print("=" * 70)

sensor_columns = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]

print(sensor_data[sensor_columns].describe())

# ============================================================
# 23. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL MISSING VALUE CHECK")
print("=" * 70)

print(sensor_data.isnull().sum())

# ============================================================
# 24. FINAL DUPLICATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DUPLICATE CHECK")
print("=" * 70)

duplicate_count = sensor_data.duplicated(subset=["subject_id", "activity", "timestamp"]).sum()

print("Duplicate synchronized rows:", duplicate_count)

# ============================================================
# 25. TIMESTAMP ORDER CHECK
# ============================================================

print("\n" + "=" * 70)
print("TIMESTAMP ORDER CHECK")
print("=" * 70)

timestamp_order_check = sensor_data.groupby(["subject_id", "activity"])["timestamp"].apply(lambda x: x.is_monotonic_increasing)

print("Groups with correct timestamp ordering:", timestamp_order_check.sum())
print("Groups with incorrect timestamp ordering:", (~timestamp_order_check).sum())

# ============================================================
# 26. FINAL DATA STRUCTURE (pre-processing checkpoint)
# ============================================================

print("\n" + "=" * 70)
print("FINAL 5-CLASS SENSOR DATA (BEFORE MISSING/NOISE/NORM/SPLIT)")
print("=" * 70)

print("Shape:", sensor_data.shape)

print("\nColumns:")
print(sensor_data.columns.tolist())

print("\nFirst 10 rows:")
print(sensor_data.head(10))

# ============================================================
# 27. SAVE ACCELEROMETER RAW DATA
# ============================================================

accel_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "accelerometer_phone_raw.csv"), index=False)

# ============================================================
# 28. SAVE GYROSCOPE RAW DATA
# ============================================================

gyro_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "gyroscope_phone_raw.csv"), index=False)

# ============================================================
# 29. SAVE FINAL 5-CLASS SYNCHRONIZED DATA (pre-processing)
# ============================================================

sensor_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "phone_sensor_5class_synchronized.csv"), index=False)

# ============================================================
# 30. SUMMARY BEFORE PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("DATASET SUMMARY BEFORE MISSING/NOISE/NORM/SPLIT")
print("=" * 70)

print("Dataset              : WISDM")
print("Sensor type          : Smartphone")
print("Sensors              : Accelerometer + Gyroscope")
print("Sensor channels      : 6")
print("Number of classes    :", sensor_data["activity_name"].nunique())
print("Number of subjects   :", sensor_data["subject_id"].nunique())
print("Total records        :", len(sensor_data))
print("Sampling rate        : Approximately 20 Hz (nominal; actual polling")
print("                        interval can vary when the device/processor")
print("                        is busy, per the WISDM documentation)")
print("Selected activities  :", selected_activities)

print("\nClass distribution:")
print(sensor_data["activity_name"].value_counts())

# ============================================================
# 31. STEP 5 - MISSING-VALUE HANDLING (MEDIAN IMPUTATION)
# ============================================================
# NOTE: Rows with missing sensor values were already dropped in
# load_txt_file() (dropna on load), so at this point sensor_data
# should contain zero NaNs in sensor_columns. This step is kept
# as an explicit, documented safeguard so the pipeline still behaves
# correctly if a future data source contains missing readings.
# This is OUR proposed preprocessing choice, not part of the
# original WISDM dataset generation procedure, and should be
# described as such in the paper/methodology section.

print("\n" + "=" * 70)
print("STEP 5: MISSING-VALUE HANDLING (MEDIAN IMPUTATION)")
print("=" * 70)

print("Missing values BEFORE imputation:")
print(sensor_data[sensor_columns].isnull().sum())

for col in sensor_columns:
    median_value = sensor_data[col].median()
    n_missing = sensor_data[col].isnull().sum()
    if n_missing > 0:
        sensor_data[col] = sensor_data[col].fillna(median_value)
    print(f"Column: {col:10s} | Missing filled: {n_missing:6d} | Median used: {median_value:.4f}")

print("\nMissing values AFTER imputation:")
print(sensor_data[sensor_columns].isnull().sum())

# ============================================================
# 32. STEP 6 - NOISE REMOVAL (MOVING AVERAGE FILTER)
# ============================================================
# Applied per subject AND per activity so the rolling window never
# mixes samples across a subject/activity boundary. The WISDM docs
# report an approximate 20 Hz polling rate, but actual polling can
# be delayed when the device/processor is busy, so timestamps are
# NOT assumed to be perfectly equally spaced; the filter is applied
# purely on sample order within each (subject_id, activity) group,
# not on elapsed time.

print("\n" + "=" * 70)
print("STEP 6: NOISE REMOVAL (MOVING AVERAGE FILTER)")
print("=" * 70)

window = 3

for col in sensor_columns:
    sensor_data[col] = (
        sensor_data.groupby(["subject_id", "activity"])[col]
        .transform(lambda x: x.rolling(window, center=True, min_periods=1).mean())
    )

print(f"Moving average filter applied (window={window}, per subject_id + activity group).")
print("min_periods=1 used so the first/last sample of each group is not turned into NaN.")

print("\nMissing values AFTER smoothing:")
print(sensor_data[sensor_columns].isnull().sum())

# ============================================================
# 33. STEP 8 (PART 1) - SUBJECT-INDEPENDENT TRAIN/VAL/TEST SPLIT
# ============================================================
# Splitting is done BEFORE normalization so the scaler can be fit
# only on the training subjects (see Step 7 below) and avoid any
# leakage from validation/test subjects into the fitted scaler.
#
# A subject appearing in training must not appear in validation
# or test.

print("\n" + "=" * 70)
print("STEP 8: SUBJECT-INDEPENDENT TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

all_subjects = sensor_data["subject_id"].unique()
all_subjects = np.sort(all_subjects)
rng.shuffle(all_subjects)

n_subjects = len(all_subjects)
n_train = int(round(0.70 * n_subjects))
n_val = int(round(0.15 * n_subjects))
# whatever remains goes to test, so rounding never drops a subject
n_test = n_subjects - n_train - n_val

train_subjects = all_subjects[:n_train]
val_subjects = all_subjects[n_train:n_train + n_val]
test_subjects = all_subjects[n_train + n_val:]

print("Total subjects   :", n_subjects)
print("Train subjects   :", len(train_subjects), "->", sorted(train_subjects.tolist()))
print("Validation subj. :", len(val_subjects), "->", sorted(val_subjects.tolist()))
print("Test subjects    :", len(test_subjects), "->", sorted(test_subjects.tolist()))

# sanity check: no overlap between subject sets
assert set(train_subjects).isdisjoint(val_subjects)
assert set(train_subjects).isdisjoint(test_subjects)
assert set(val_subjects).isdisjoint(test_subjects)
print("Subject overlap check passed: no subject appears in more than one split.")

train_data = sensor_data[sensor_data["subject_id"].isin(train_subjects)].copy()
val_data = sensor_data[sensor_data["subject_id"].isin(val_subjects)].copy()
test_data = sensor_data[sensor_data["subject_id"].isin(test_subjects)].copy()

print("\nRow counts:")
print("Train rows      :", len(train_data))
print("Validation rows :", len(val_data))
print("Test rows       :", len(test_data))

print("\nClass distribution (train):")
print(train_data["activity_name"].value_counts())
print("\nClass distribution (validation):")
print(val_data["activity_name"].value_counts())
print("\nClass distribution (test):")
print(test_data["activity_name"].value_counts())

# ============================================================
# 34. STEP 7 - NORMALIZATION (MIN-MAX SCALING, FIT ON TRAIN ONLY)
# ============================================================
# The scaler is fit ONLY on the training split and then used to
# transform validation and test. The scaler is never re-fit on
# validation/test data, to avoid data leakage.

print("\n" + "=" * 70)
print("STEP 7: NORMALIZATION (MIN-MAX SCALING)")
print("=" * 70)

scaler = MinMaxScaler()

train_data[sensor_columns] = scaler.fit_transform(train_data[sensor_columns])
val_data[sensor_columns] = scaler.transform(val_data[sensor_columns])
test_data[sensor_columns] = scaler.transform(test_data[sensor_columns])

print("MinMaxScaler fit on TRAIN data only.")
print("Same fitted scaler used to transform validation and test data.")

print("\nTrain sensor ranges after scaling:")
print(train_data[sensor_columns].describe().loc[["min", "max"]])

# ============================================================
# 35. SAVE PREPROCESSED SPLITS
# ============================================================

train_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "phone_sensor_5class_train.csv"), index=False)
val_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "phone_sensor_5class_val.csv"), index=False)
test_data.to_csv(os.path.join(CSV_OUTPUT_DIR, "phone_sensor_5class_test.csv"), index=False)

# ============================================================
# 36. STEP 9 - WINDOWING (SLIDING WINDOW, EXPLICIT OVERLAP)
# ============================================================
# WISDM's OFFICIAL transformed data uses 10-second, NON-OVERLAPPING
# segments of 200 readings (20 Hz x 10 sec = 200 samples/window).
#
# This pipeline instead uses a SLIDING window with an EXPLICITLY
# DEFINED overlap, which is a deliberate, documented departure from
# the official WISDM transformation - not a reproduction of it.
#
#   Sampling rate = 20 Hz
#   Window length = 10 sec  -> WINDOW_SIZE   = 200 samples
#   Overlap       = 50%     -> STEP_SIZE     = 100 samples
#
# Set OVERLAP = 0.0 (STEP_SIZE = WINDOW_SIZE) to instead reproduce
# the official non-overlapping WISDM segmentation.
#
# Windowing is performed SEPARATELY on train_data, val_data and
# test_data (each already restricted to its own, non-overlapping
# set of subjects), and within each split, windows are built per
# (subject_id, activity) group sorted by timestamp so a window
# never mixes two different subjects or two different activities.

print("\n" + "=" * 70)
print("STEP 9: WINDOWING (SLIDING WINDOW WITH EXPLICIT OVERLAP)")
print("=" * 70)

SAMPLING_RATE_HZ = 20
WINDOW_SECONDS = 10
WINDOW_SIZE = SAMPLING_RATE_HZ * WINDOW_SECONDS  # 200 samples

OVERLAP = 0.50  # 50% overlap; set to 0.0 for non-overlapping WISDM-style windows
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 100 samples at 50% overlap

print("Sampling rate      :", SAMPLING_RATE_HZ, "Hz")
print("Window length      :", WINDOW_SECONDS, "sec")
print("Samples per window :", WINDOW_SIZE)
print("Overlap            :", OVERLAP * 100, "%")
print("Step size          :", STEP_SIZE, "samples")
print("Windowing strategy : SLIDING WINDOW (own strategy, distinct from")
print("                      the official WISDM non-overlapping segmentation)")

def create_windows(df, sensor_columns, window_size, step_size):
    """
    Slide a fixed-size window over each (subject_id, activity) group,
    sorted by timestamp, producing:
      X : array of shape (n_windows, window_size, n_channels)
      y : array of shape (n_windows,)         -> activity_name label
      subj : array of shape (n_windows,)      -> subject_id for that window
    A window is only kept if it is completely filled (no partial
    windows at the end of a group).
    """
    X_list = []
    y_list = []
    subj_list = []

    grouped = df.sort_values("timestamp").groupby(["subject_id", "activity", "activity_name"])

    for (subject_id, activity, activity_name), group in grouped:
        values = group[sensor_columns].to_numpy()
        n_samples = len(values)

        if n_samples < window_size:
            continue

        for start in range(0, n_samples - window_size + 1, step_size):
            end = start + window_size
            window_values = values[start:end]
            X_list.append(window_values)
            y_list.append(activity_name)
            subj_list.append(subject_id)

    if len(X_list) == 0:
        X = np.empty((0, window_size, len(sensor_columns)))
        y = np.empty((0,), dtype=object)
        subj = np.empty((0,))
        return X, y, subj

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    subj = np.array(subj_list)

    return X, y, subj

X_train, y_train, subj_train = create_windows(train_data, sensor_columns, WINDOW_SIZE, STEP_SIZE)
X_val, y_val, subj_val = create_windows(val_data, sensor_columns, WINDOW_SIZE, STEP_SIZE)
X_test, y_test, subj_test = create_windows(test_data, sensor_columns, WINDOW_SIZE, STEP_SIZE)

print("\nWindowed shapes (n_windows, timesteps, channels):")
print("X_train:", X_train.shape)
print("X_val  :", X_val.shape)
print("X_test :", X_test.shape)

print("\nWindow label distribution (train):")
print(pd.Series(y_train).value_counts())
print("\nWindow label distribution (validation):")
print(pd.Series(y_val).value_counts())
print("\nWindow label distribution (test):")
print(pd.Series(y_test).value_counts())

# sanity check: every window's subject must belong to its intended split
assert set(np.unique(subj_train)).issubset(set(train_subjects))
assert set(np.unique(subj_val)).issubset(set(val_subjects))
assert set(np.unique(subj_test)).issubset(set(test_subjects))
print("\nSubject-consistency check passed: every window's subject_id")
print("belongs only to its own split (train/val/test).")

# ============================================================
# 36b. LIMIT DATA: MAX N WINDOWS PER CLASS (SPEED / MEMORY CONTROL)
# ============================================================
# Capping the number of windows per class keeps training fast while
# still giving every class a reasonably large, balanced sample. This
# is a practical experiment-size choice, not a WISDM-specified step.

MAX_PER_CLASS = 1000
CAP_SEED = 42

def subsample_per_class(X, y, subj, max_per_class, seed):
    rng_cap = np.random.default_rng(seed)
    keep_idx = []

    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        if len(label_idx) > max_per_class:
            label_idx = rng_cap.choice(label_idx, size=max_per_class, replace=False)
        keep_idx.append(label_idx)

    keep_idx = np.concatenate(keep_idx)
    keep_idx = np.sort(keep_idx)

    return X[keep_idx], y[keep_idx], subj[keep_idx]

print("\n" + "=" * 70)
print(f"LIMITING DATA TO AT MOST {MAX_PER_CLASS} WINDOWS PER CLASS")
print("=" * 70)

print("Before capping:")
print("Train:", pd.Series(y_train).value_counts().to_dict())
print("Val  :", pd.Series(y_val).value_counts().to_dict())
print("Test :", pd.Series(y_test).value_counts().to_dict())

X_train, y_train, subj_train = subsample_per_class(X_train, y_train, subj_train, MAX_PER_CLASS, CAP_SEED)
X_val, y_val, subj_val = subsample_per_class(X_val, y_val, subj_val, MAX_PER_CLASS, CAP_SEED)
X_test, y_test, subj_test = subsample_per_class(X_test, y_test, subj_test, MAX_PER_CLASS, CAP_SEED)

print("\nAfter capping:")
print("Train:", pd.Series(y_train).value_counts().to_dict(), "| shape:", X_train.shape)
print("Val  :", pd.Series(y_val).value_counts().to_dict(), "| shape:", X_val.shape)
print("Test :", pd.Series(y_test).value_counts().to_dict(), "| shape:", X_test.shape)

# ============================================================
# 37. SAVE WINDOWED ARRAYS
# ============================================================

np.save(os.path.join(NPY_OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(NPY_OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(NPY_OUTPUT_DIR, "subj_train.npy"), subj_train)

np.save(os.path.join(NPY_OUTPUT_DIR, "X_val.npy"), X_val)
np.save(os.path.join(NPY_OUTPUT_DIR, "y_val.npy"), y_val)
np.save(os.path.join(NPY_OUTPUT_DIR, "subj_val.npy"), subj_val)

np.save(os.path.join(NPY_OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(NPY_OUTPUT_DIR, "y_test.npy"), y_test)
np.save(os.path.join(NPY_OUTPUT_DIR, "subj_test.npy"), subj_test)

# ============================================================
# 38. STEP 10 - FEATURE EXTRACTION (HANDCRAFTED FEATURES)
# ============================================================


print("\n" + "=" * 70)
print("STEP 10: FEATURE EXTRACTION (HANDCRAFTED FEATURES)")
print("=" * 70)

N_HIST_BINS = 10

def extract_window_features(window, channel_names):
    """
    window       : array of shape (window_size, n_channels)
    channel_names: list of channel names matching window's columns

    Returns a dict of {feature_name: value} for this single window.
    """
    features = {}

    for idx, ch_name in enumerate(channel_names):
        x = window[:, idx]

        mean_val = np.mean(x)
        std_val = np.std(x)
        var_val = np.var(x)
        avg_abs_diff = np.mean(np.abs(np.diff(x))) if len(x) > 1 else 0.0

        peaks, _ = find_peaks(x)
        peak_count = len(peaks)

        euclidean_norm_mean = np.mean(np.sqrt(x ** 2))

        hist_counts, _ = np.histogram(x, bins=N_HIST_BINS)
        hist_counts = hist_counts / hist_counts.sum() if hist_counts.sum() > 0 else hist_counts

        features[f"{ch_name}_mean"] = mean_val
        features[f"{ch_name}_std"] = std_val
        features[f"{ch_name}_var"] = var_val
        features[f"{ch_name}_avg_abs_diff"] = avg_abs_diff
        features[f"{ch_name}_peak_count"] = peak_count
        features[f"{ch_name}_euclidean_norm_mean"] = euclidean_norm_mean

        for b in range(N_HIST_BINS):
            features[f"{ch_name}_hist_bin{b}"] = hist_counts[b]

    return features

def build_feature_dataframe(X, y, subj, sensor_columns):
    """
    X    : array (n_windows, window_size, n_channels) from create_windows()
    y    : array (n_windows,) activity_name labels
    subj : array (n_windows,) subject_id per window

    Adds accel_mag / gyro_mag as extra derived channels, then
    extracts handcrafted features per window.
    Returns a single DataFrame: one row per window.
    """
    accel_idx = [sensor_columns.index(c) for c in ["accel_x", "accel_y", "accel_z"]]
    gyro_idx = [sensor_columns.index(c) for c in ["gyro_x", "gyro_y", "gyro_z"]]

    channel_names = list(sensor_columns) + ["accel_mag", "gyro_mag"]

    rows = []
    for i in range(X.shape[0]):
        window = X[i]

        accel_mag = np.sqrt(np.sum(window[:, accel_idx] ** 2, axis=1))
        gyro_mag = np.sqrt(np.sum(window[:, gyro_idx] ** 2, axis=1))

        full_window = np.column_stack([window, accel_mag, gyro_mag])

        row_features = extract_window_features(full_window, channel_names)
        row_features["activity_name"] = y[i]
        row_features["subject_id"] = subj[i]

        rows.append(row_features)

    return pd.DataFrame(rows)

train_features = build_feature_dataframe(X_train, y_train, subj_train, sensor_columns)
val_features = build_feature_dataframe(X_val, y_val, subj_val, sensor_columns)
test_features = build_feature_dataframe(X_test, y_test, subj_test, sensor_columns)

print("Train feature matrix shape:", train_features.shape)
print("Val   feature matrix shape:", val_features.shape)
print("Test  feature matrix shape:", test_features.shape)

print("\nFeature columns (sample):")
print(train_features.columns.tolist()[:12], "...")

# save handcrafted feature tables alongside the other CSV outputs
train_features.to_csv(os.path.join(CSV_OUTPUT_DIR, "features_train.csv"), index=False)
val_features.to_csv(os.path.join(CSV_OUTPUT_DIR, "features_val.csv"), index=False)
test_features.to_csv(os.path.join(CSV_OUTPUT_DIR, "features_test.csv"), index=False)

# also save as .npy (features-only, labels separately) for ML pipelines
feature_cols = [c for c in train_features.columns if c not in ("activity_name", "subject_id")]

np.save(os.path.join(NPY_OUTPUT_DIR, "features_train.npy"), train_features[feature_cols].to_numpy())
np.save(os.path.join(NPY_OUTPUT_DIR, "features_val.npy"), val_features[feature_cols].to_numpy())
np.save(os.path.join(NPY_OUTPUT_DIR, "features_test.npy"), test_features[feature_cols].to_numpy())

np.save(os.path.join(NPY_OUTPUT_DIR, "features_y_train.npy"), train_features["activity_name"].to_numpy())
np.save(os.path.join(NPY_OUTPUT_DIR, "features_y_val.npy"), val_features["activity_name"].to_numpy())
np.save(os.path.join(NPY_OUTPUT_DIR, "features_y_test.npy"), test_features["activity_name"].to_numpy())

# ============================================================
# 39. FINAL PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 70)

print("Dataset               : WISDM")
print("Sensor type           : Smartphone")
print("Sensors               : Accelerometer + Gyroscope")
print("Sensor channels       : 6")
print("Missing-value method  : Median imputation (our preprocessing choice)")
print("Noise-removal method  : Moving average filter (window =", window, ", per subject+activity)")
print("Normalization method  : Min-Max scaling (fit on train only)")
print("Split strategy        : Subject-independent")
print("  Train subjects      :", len(train_subjects), "| rows:", len(train_data))
print("  Validation subjects :", len(val_subjects), "| rows:", len(val_data))
print("  Test subjects       :", len(test_subjects), "| rows:", len(test_data))
print("Windowing strategy    : Sliding window (own strategy, not official WISDM)")
print("  Window size          :", WINDOW_SIZE, "samples (", WINDOW_SECONDS, "sec @", SAMPLING_RATE_HZ, "Hz )")
print("  Overlap              :", OVERLAP * 100, "% | Step size:", STEP_SIZE, "samples")
print("  X_train shape        :", X_train.shape)
print("  X_val shape          :", X_val.shape)
print("  X_test shape         :", X_test.shape)
print("Feature extraction    : Mean, STD, Variance, Avg Abs Diff, Peak count,")
print("                        Euclidean norm (mean), Histogram bins x", N_HIST_BINS)
print("  Channels             :", sensor_columns, "+ ['accel_mag', 'gyro_mag']")
print("  Train features shape :", train_features.shape)
print("  Val features shape   :", val_features.shape)
print("  Test features shape  :", test_features.shape)
print("Selected activities   :", selected_activities)

print("\nOutput folders:")
print("CSV outputs -> ./" + CSV_OUTPUT_DIR + "/")
print("NPY outputs -> ./" + NPY_OUTPUT_DIR + "/")

print("\nSaved CSV files (in", CSV_OUTPUT_DIR, "):")
print("accelerometer_phone_raw.csv")
print("gyroscope_phone_raw.csv")
print("phone_sensor_5class_synchronized.csv  (pre-processing)")
print("phone_sensor_5class_train.csv")
print("phone_sensor_5class_val.csv")
print("phone_sensor_5class_test.csv")
print("features_train.csv")
print("features_val.csv")
print("features_test.csv")

print("\nSaved NPY files (in", NPY_OUTPUT_DIR, "):")
print("X_train.npy / y_train.npy / subj_train.npy")
print("X_val.npy   / y_val.npy   / subj_val.npy")
print("X_test.npy  / y_test.npy  / subj_test.npy")
print("features_train.npy / features_y_train.npy")
print("features_val.npy   / features_y_val.npy")
print("features_test.npy  / features_y_test.npy")

# ============================================================
# 40. STEP 11/12 - ResCBAR-FusionNet (TWO-STREAM ARCHITECTURE)
# ============================================================
# Stream A (sequence branch):  raw (200, 6) window -> 1D-CNN -> BiGRU -> CBAM
# Stream B (statistical branch): handcrafted feature vector -> Dense
# The two streams are concatenated ("Feature Fusion"), passed through
# a residual dense block, then a final Dense + Softmax classifier.
#
# NOTE ON CLASS COUNT: Steps 1-10 of this pipeline were built around
# the 5 SELECTED activities (selected_activities). Step 15 of the
# methodology mentions "18-class prediction" (the full WISDM activity
# set). This code stays consistent with the actual data produced by
# this pipeline and infers num_classes automatically from whatever
# labels are present in y_train, so it works unchanged whether you
# keep the 5-class subset or later regenerate the pipeline on all 18
# activities. It does NOT fabricate an 18-class output on 5-class data.

# Reduce TensorFlow log noise and avoid a known spurious Grappler/MLIR
# warning ("... has an empty op name ...") that can appear when models
# are repeatedly built/rebuilt (e.g. during the POA search below). This
# does not affect training correctness - it only silences a diagnostic
# graph-optimizer log line. Must be set BEFORE importing tensorflow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DISABLE_MLIR_GRAPH_OPTIMIZATION"] = "1"

import tensorflow as tf
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.utils import to_categorical

tf.get_logger().setLevel("ERROR")
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 70)
print("STEP 11/12: ResCBAR-FusionNet ARCHITECTURE")
print("=" * 70)

OUTPUT_PLOTS_DIR = "outputs_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# ---- label encoding (shared across train/val/test) ----
label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train_enc = to_categorical(label_encoder.transform(y_train))
y_val_enc = to_categorical(label_encoder.transform(y_val))
y_test_enc = to_categorical(label_encoder.transform(y_test))

NUM_CLASSES = y_train_enc.shape[1]
CLASS_NAMES = list(label_encoder.classes_)

print("Classes detected from data:", CLASS_NAMES)
print("Number of classes         :", NUM_CLASSES)

# ---- inputs for the two streams ----
SEQ_SHAPE = X_train.shape[1:]                       # (200, 6)
FEAT_DIM = train_features[feature_cols].shape[1]    # handcrafted feature vector length

X_train_feat = train_features[feature_cols].to_numpy()
X_val_feat = val_features[feature_cols].to_numpy()
X_test_feat = test_features[feature_cols].to_numpy()


def cbam_block(input_feature, ratio=8):
    """
    Convolutional Block Attention Module, 1D version.
    Channel attention (what to focus on) followed by
    spatial/temporal attention (where in the sequence to focus).
    """
    channels = input_feature.shape[-1]

    # ---- channel attention ----
    avg_pool = layers.GlobalAveragePooling1D()(input_feature)
    max_pool = layers.GlobalMaxPooling1D()(input_feature)

    shared_dense_1 = layers.Dense(max(channels // ratio, 1), activation="relu")
    shared_dense_2 = layers.Dense(channels, activation=None)

    avg_out = shared_dense_2(shared_dense_1(avg_pool))
    max_out = shared_dense_2(shared_dense_1(max_pool))

    channel_attention = layers.Activation("sigmoid")(layers.Add()([avg_out, max_out]))
    channel_attention = layers.Reshape((1, channels))(channel_attention)
    x = layers.Multiply()([input_feature, channel_attention])

    # ---- spatial (temporal) attention ----
    avg_pool_spatial = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool_spatial = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])

    spatial_attention = layers.Conv1D(1, kernel_size=7, padding="same", activation="sigmoid")(concat)
    x = layers.Multiply()([x, spatial_attention])

    return x


def build_rescbar_fusionnet(seq_shape, feat_dim, num_classes, hp):
    """
    hp: dict with keys
        conv_filters, kernel_size, gru_units, dense_units,
        dropout, learning_rate
    """
    # Start from a clean graph/naming space every time this is called.
    # Building many models back-to-back (as POA does) without this can
    # leave stale layers/names around and trigger a spurious Grappler
    # "empty op name" warning; clearing first avoids it entirely.
    tf.keras.backend.clear_session()

    # L2 weight regularization applied throughout - this is what most
    # directly shrinks the train/val gap you saw (train ~0.97, val
    # ~0.6-0.75): the network was fitting subject-specific noise in
    # the training subjects that doesn't transfer to unseen subjects.
    l2 = regularizers.l2(1e-4)
    # dropout is floored at 0.3 regardless of what a given hp dict
    # requests, since very low dropout was part of why training
    # accuracy raced ahead of validation accuracy
    dropout_rate = max(hp["dropout"], 0.3)

    # ---- Stream A: raw sequence -> 1D-CNN -> BiGRU -> CBAM (+ residual) ----
    seq_input = Input(shape=seq_shape, name="sequence_input")

    conv1 = layers.Conv1D(hp["conv_filters"], hp["kernel_size"], padding="same",
                           activation="relu", kernel_regularizer=l2)(seq_input)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.SpatialDropout1D(0.2)(conv1)

    conv2 = layers.Conv1D(hp["conv_filters"], hp["kernel_size"], padding="same",
                           activation="relu", kernel_regularizer=l2)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.SpatialDropout1D(0.2)(conv2)

    bigru = layers.Bidirectional(
        layers.GRU(hp["gru_units"], return_sequences=True,
                    kernel_regularizer=l2, recurrent_dropout=0.2)
    )(conv2)

    attended = cbam_block(bigru)

    # residual connection: project conv2 to match bigru+CBAM's channel dim,
    # then add, so information can skip the BiGRU/CBAM path
    residual_proj = layers.Conv1D(attended.shape[-1], kernel_size=1, padding="same")(conv2)
    fused_seq = layers.Add(name="residual_add")([attended, residual_proj])
    fused_seq = layers.Activation("relu")(fused_seq)

    seq_repr = layers.GlobalAveragePooling1D()(fused_seq)
    seq_repr = layers.Dropout(dropout_rate)(seq_repr)

    # ---- Stream B: handcrafted statistical features -> Dense ----
    feat_input = Input(shape=(feat_dim,), name="feature_input")
    feat_repr = layers.Dense(hp["dense_units"], activation="relu",
                              kernel_regularizer=l2)(feat_input)
    feat_repr = layers.BatchNormalization()(feat_repr)
    feat_repr = layers.Dropout(dropout_rate)(feat_repr)

    # ---- Feature Fusion ----
    fused = layers.Concatenate(name="feature_fusion")([seq_repr, feat_repr])

    # ---- Residual dense block after fusion ----
    dense_in = layers.Dense(hp["dense_units"], activation="relu",
                             kernel_regularizer=l2)(fused)
    dense_out = layers.Dense(hp["dense_units"], activation="relu",
                              kernel_regularizer=l2)(dense_in)
    dense_proj = layers.Dense(hp["dense_units"])(fused)  # projection to match shapes
    residual_dense = layers.Add()([dense_out, dense_proj])
    residual_dense = layers.Activation("relu")(residual_dense)
    residual_dense = layers.Dropout(dropout_rate)(residual_dense)

    output = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=l2)(residual_dense)

    model = models.Model(inputs=[seq_input, feat_input], outputs=output,
                          name="ResCBAR_FusionNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


DEFAULT_HP = {
    "conv_filters": 64,
    "kernel_size": 5,
    "gru_units": 64,
    "dense_units": 128,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
}

demo_model = build_rescbar_fusionnet(SEQ_SHAPE, FEAT_DIM, NUM_CLASSES, DEFAULT_HP)
demo_model.summary()

# ============================================================
# 41. STEP 13/14 - POA (PELICAN OPTIMIZATION ALGORITHM) SEARCH
# ============================================================
# Search space is EXPLICITLY our own experimental choice (Step 13),
# not a dataset property.
#
# WHY THE SEARCH WAS "STUCK": every candidate trains a full BiGRU
# model from scratch. With POP_SIZE=5, N_ITER=5 and 2 phases/pelican,
# that is 5 + 5*5*2 = 55 model trainings - on the FULL capped dataset
# (up to 1000 windows/class), each with no visible progress printed
# until a whole generation finished. That is genuinely slow on CPU,
# and it also LOOKED stuck because nothing printed for minutes at a
# time. Two fixes are applied below:
#   1. The search itself now runs on a SMALLER, separately-capped
#      subsample (POA_SAMPLE_PER_CLASS) - this only affects which
#      hyperparameters get chosen, NOT the final model, which is
#      still trained on the full capped dataset in Step 15.
#   2. Every single candidate now prints its hyperparameters, fitness,
#      and wall-clock time as soon as it finishes, so progress is
#      visible in real time instead of going silent for minutes.

print("\n" + "=" * 70)
print("STEP 13/14: POA HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

SEARCH_SPACE = {
    "learning_rate": (1e-4, 1e-2),
    "conv_filters": (16, 96),
    "kernel_size": (3, 7),
    "dense_units": (32, 128),   # capped lower than before (was 256) - large dense
                                # layers were a major source of the overfitting gap
    "dropout": (0.3, 0.5),      # floor raised from 0.1 to 0.3 for the same reason
    "batch_size": (16, 128),
}
PARAM_NAMES = list(SEARCH_SPACE.keys())
INT_PARAMS = {"conv_filters", "kernel_size", "dense_units", "batch_size"}

POP_SIZE = 4        # number of Pelicans; increase (e.g. 15-20) for a real run
N_ITER = 3          # POA generations; increase (e.g. 20-30) for a real run
SEARCH_EPOCHS = 3   # max epochs per candidate during search only
SEARCH_PATIENCE = 2 # early-stop a candidate quickly if it isn't improving

# The search runs on a SMALLER subsample so each candidate trains fast.
# This does NOT change the data used for final training/testing later.
POA_SAMPLE_PER_CLASS = 150
X_train_search, y_train_search, _ = subsample_per_class(
    X_train, y_train, subj_train, POA_SAMPLE_PER_CLASS, CAP_SEED
)
X_val_search, y_val_search, _ = subsample_per_class(
    X_val, y_val, subj_val, POA_SAMPLE_PER_CLASS, CAP_SEED
)

# handcrafted feature rows must line up 1:1 with the (already capped)
# X_train / X_val arrays, so re-derive the matching feature rows by
# recomputing them straight from the search-sampled windows.
train_features_search = build_feature_dataframe(
    X_train_search, y_train_search, np.zeros(len(y_train_search)), sensor_columns
)
val_features_search = build_feature_dataframe(
    X_val_search, y_val_search, np.zeros(len(y_val_search)), sensor_columns
)
X_train_feat_search = train_features_search[feature_cols].to_numpy()
X_val_feat_search = val_features_search[feature_cols].to_numpy()

y_train_enc_search = to_categorical(label_encoder.transform(y_train_search), num_classes=NUM_CLASSES)
y_val_enc_search = to_categorical(label_encoder.transform(y_val_search), num_classes=NUM_CLASSES)

n_total_evals = POP_SIZE + N_ITER * POP_SIZE * 2
print(f"Search sample size : {len(y_train_search)} train / {len(y_val_search)} val windows")
print(f"Population size    : {POP_SIZE} | Generations: {N_ITER} | Epochs/candidate: {SEARCH_EPOCHS}")
print(f"Total candidate evaluations this run: up to {n_total_evals}")
print("(Progress for every single candidate is printed live below.)\n")

rng_poa = np.random.default_rng(RANDOM_SEED)
eval_counter = {"n": 0}


def clip_to_bounds(vec):
    clipped = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = SEARCH_SPACE[name]
        val = np.clip(vec[i], low, high)
        if name in INT_PARAMS:
            val = int(round(val))
        clipped[name] = val
    return clipped


def hp_to_vector(hp):
    return np.array([hp[name] for name in PARAM_NAMES], dtype=float)


def random_candidate():
    vec = np.array([
        rng_poa.uniform(SEARCH_SPACE[name][0], SEARCH_SPACE[name][1])
        for name in PARAM_NAMES
    ])
    return vec


def fitness_function(hp, tag=""):
    """Fitness = 1 - F1(validation, on the small search subsample). Lower is better."""
    eval_counter["n"] += 1
    t0 = time.time()

    hp_full = dict(hp)
    hp_full["gru_units"] = 64  # kept fixed (not part of Step 13's search space)

    model = build_rescbar_fusionnet(SEQ_SHAPE, FEAT_DIM, NUM_CLASSES, hp_full)

    search_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=SEARCH_PATIENCE, restore_best_weights=True
    )

    model.fit(
        [X_train_search, X_train_feat_search], y_train_enc_search,
        validation_data=([X_val_search, X_val_feat_search], y_val_enc_search),
        epochs=SEARCH_EPOCHS,
        batch_size=hp_full["batch_size"],
        callbacks=[search_early_stop],
        verbose=0
    )

    val_pred = np.argmax(model.predict([X_val_search, X_val_feat_search], verbose=0), axis=1)
    val_true = np.argmax(y_val_enc_search, axis=1)
    f1_val = f1_score(val_true, val_pred, average="weighted")

    tf.keras.backend.clear_session()
    elapsed = time.time() - t0
    fitness = 1.0 - f1_val

    print(f"  [Eval {eval_counter['n']}/{n_total_evals}]{tag} "
          f"fitness={fitness:.4f} (F1={f1_val:.4f}) | {elapsed:.1f}s | hp={hp_full}")

    return fitness


poa_start_time = time.time()

# ---- initialize Pelican population ----
print("Evaluating initial population...")
population = np.array([random_candidate() for _ in range(POP_SIZE)])
fitness_values = np.array([
    fitness_function(clip_to_bounds(ind), tag=f" [init {i + 1}/{POP_SIZE}]")
    for i, ind in enumerate(population)
])

best_idx = np.argmin(fitness_values)
best_hp = clip_to_bounds(population[best_idx])
best_fitness = fitness_values[best_idx]

print(f"\nInitial best fitness (1 - F1_val): {best_fitness:.4f}")
print("Initial best hyperparameters:", best_hp)

for t in range(N_ITER):
    print(f"\n--- Generation {t + 1}/{N_ITER} ---")
    for i in range(POP_SIZE):
        # ---- Phase 1: moving towards the prey (best pelican) - exploration ----
        I = rng_poa.choice([1, 2])
        new_pos_p1 = population[i] + rng_poa.uniform() * (population[best_idx] - I * population[i])
        new_hp_p1 = clip_to_bounds(new_pos_p1)
        new_fit_p1 = fitness_function(new_hp_p1, tag=f" [gen {t + 1} pelican {i + 1} phase1]")
        if new_fit_p1 < fitness_values[i]:
            population[i] = hp_to_vector(new_hp_p1)
            fitness_values[i] = new_fit_p1

        # ---- Phase 2: winging on the water surface - exploitation ----
        R = 0.2
        new_pos_p2 = population[i] + R * (1 - (t + 1) / N_ITER) * (2 * rng_poa.uniform() - 1) * population[i]
        new_hp_p2 = clip_to_bounds(new_pos_p2)
        new_fit_p2 = fitness_function(new_hp_p2, tag=f" [gen {t + 1} pelican {i + 1} phase2]")
        if new_fit_p2 < fitness_values[i]:
            population[i] = hp_to_vector(new_hp_p2)
            fitness_values[i] = new_fit_p2

    gen_best_idx = np.argmin(fitness_values)
    if fitness_values[gen_best_idx] < best_fitness:
        best_idx = gen_best_idx
        best_fitness = fitness_values[gen_best_idx]
        best_hp = clip_to_bounds(population[best_idx])

    elapsed_total = time.time() - poa_start_time
    print(f"Generation {t + 1}/{N_ITER} done | best fitness so far: {best_fitness:.4f} "
          f"| elapsed: {elapsed_total / 60:.1f} min")

best_hp["gru_units"] = 64
print("\nBest hyperparameters found by POA:")
print(best_hp)
print("Best validation fitness (1 - F1_val):", best_fitness)
print(f"Total POA search time: {(time.time() - poa_start_time) / 60:.1f} min")



# ============================================================
# 42. STEP 15 - FINAL ResCBAR-FusionNet TRAINING (20 EPOCHS)
# ============================================================

print("\n" + "=" * 70)
print("STEP 15: FINAL MODEL TRAINING WITH OPTIMIZED HYPERPARAMETERS (20 EPOCHS)")
print("=" * 70)

# Additional imports for edge deployment and enhanced plots
import sys
import gc
import tracemalloc

try:
    import psutil
except ImportError:
    import subprocess as _sp
    _sp.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil

from scipy.interpolate import make_interp_spline
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.metrics import precision_recall_curve, average_precision_score

MODEL_OUTPUT_DIR = "outputs_model"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

OUTPUT_PLOTS_DIR = "outputs_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# ---- Data augmentation on the TRAINING windows only ----
AUG_JITTER_SIGMA = 0.03
AUG_SCALE_RANGE = (0.9, 1.1)
aug_rng = np.random.default_rng(RANDOM_SEED)


def augment_window(window):
    """window: array (window_size, n_channels). Returns an augmented copy."""
    augmented = window.copy()
    channel_std = augmented.std(axis=0, keepdims=True)
    channel_std[channel_std == 0] = 1e-6
    noise = aug_rng.normal(loc=0.0, scale=AUG_JITTER_SIGMA, size=augmented.shape) * channel_std
    augmented = augmented + noise
    scale = aug_rng.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1], size=(1, augmented.shape[1]))
    augmented = augmented * scale
    return augmented


print("Augmenting training windows (jitter + random scaling)...")
X_train_aug = np.stack([augment_window(w) for w in X_train], axis=0)

aug_features_df = build_feature_dataframe(
    X_train_aug, y_train, np.zeros(len(y_train)), sensor_columns
)
X_train_feat_aug = aug_features_df[feature_cols].to_numpy()

X_train_combined = np.concatenate([X_train, X_train_aug], axis=0)
X_train_feat_combined = np.concatenate([X_train_feat, X_train_feat_aug], axis=0)
y_train_enc_combined = np.concatenate([y_train_enc, y_train_enc], axis=0)

print("Original training windows :", X_train.shape[0])
print("Augmented training windows:", X_train_aug.shape[0])
print("Combined training windows :", X_train_combined.shape[0])

final_model = build_rescbar_fusionnet(SEQ_SHAPE, FEAT_DIM, NUM_CLASSES, best_hp)

# ReduceLROnPlateau to assist convergence within 20 epochs
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

FINAL_EPOCHS = 20

history = final_model.fit(
    [X_train_combined, X_train_feat_combined], y_train_enc_combined,
    validation_data=([X_val, X_val_feat], y_val_enc),
    epochs=FINAL_EPOCHS,
    batch_size=best_hp["batch_size"],
    callbacks=[reduce_lr],
    verbose=1
)

# ---- Save Keras model ----
final_model.save(os.path.join(MODEL_OUTPUT_DIR, "rescbar_fusionnet_final.h5"))
print("Keras model saved to:", os.path.join(MODEL_OUTPUT_DIR, "rescbar_fusionnet_final.h5"))

test_pred_probs = final_model.predict([X_test, X_test_feat], verbose=0)
test_pred = np.argmax(test_pred_probs, axis=1)
test_true = np.argmax(y_test_enc, axis=1)

# ============================================================
# 43. STEP 16 - CLASSIFICATION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 16: CLASSIFICATION EVALUATION (TEST SET)")
print("=" * 70)

test_accuracy = accuracy_score(test_true, test_pred)
test_precision = precision_score(test_true, test_pred, average="weighted")
test_recall = recall_score(test_true, test_pred, average="weighted")
test_f1 = f1_score(test_true, test_pred, average="weighted")
conf_mat = confusion_matrix(test_true, test_pred)

# Per-class metrics
per_class_precision = precision_score(test_true, test_pred, average=None)
per_class_recall = recall_score(test_true, test_pred, average=None)
per_class_f1 = f1_score(test_true, test_pred, average=None)

# Specificity per class
specificity_per_class = []
for i in range(NUM_CLASSES):
    tn = np.sum(conf_mat) - np.sum(conf_mat[i, :]) - np.sum(conf_mat[:, i]) + conf_mat[i, i]
    fp = np.sum(conf_mat[:, i]) - conf_mat[i, i]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specificity_per_class.append(spec)
specificity_per_class = np.array(specificity_per_class)

# FPR and FNR per class
fpr_per_class = 1.0 - specificity_per_class
fnr_per_class = 1.0 - per_class_recall

# Overall specificity (weighted average)
class_counts_test = np.sum(conf_mat, axis=1)
overall_specificity = np.average(specificity_per_class, weights=class_counts_test)

print(f"Accuracy   : {test_accuracy:.4f}")
print(f"Precision  : {test_precision:.4f}")
print(f"Recall     : {test_recall:.4f}")
print(f"F1-score   : {test_f1:.4f}")
print(f"Specificity: {overall_specificity:.4f}")
print("\nPer-class report:")
print(classification_report(test_true, test_pred, target_names=CLASS_NAMES))

ACCURACY_TARGET = 0.95
if test_accuracy >= ACCURACY_TARGET:
    print(f"\nTest accuracy {test_accuracy:.4f} meets the {ACCURACY_TARGET:.2f} target.")
else:
    print(f"\nTest accuracy {test_accuracy:.4f} is BELOW the {ACCURACY_TARGET:.2f} target.")
    print("Consider: more POA generations/population, more training epochs,")
    print("more/better labeled data, or revisiting window/feature choices.")

# ============================================================
# 44. EDGE DEVICE DEPLOYMENT (TFLite Conversion + Inference)
# ============================================================
# The trained ResCBAR-FusionNet model is converted to TensorFlow Lite
# format for deployment on edge devices (smartphone / smartwatch).
# Two versions are produced:
#   1. Float32 (baseline, no quantization)
#   2. INT8 (post-training dynamic range quantization)
# Edge inference is simulated locally using the TFLite interpreter.

print("\n" + "=" * 70)
print("STEP 17: EDGE DEVICE DEPLOYMENT (TFLite Conversion)")
print("=" * 70)

# ---- Float32 TFLite conversion ----
print("Converting model to TFLite (Float32)...")
converter_float = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_float_model = converter_float.convert()

tflite_float_path = os.path.join(MODEL_OUTPUT_DIR, "rescbar_fusionnet_float32.tflite")
with open(tflite_float_path, "wb") as f:
    f.write(tflite_float_model)
print(f"Float32 TFLite model saved: {tflite_float_path}")

# ---- INT8 Quantized TFLite conversion ----
print("Converting model to TFLite (INT8 Quantized)...")
converter_quant = tf.lite.TFLiteConverter.from_keras_model(final_model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_dataset_gen():
    """Yields representative samples for calibration during quantization."""
    n_cal = min(200, len(X_train))
    for i in range(n_cal):
        yield [
            X_train[i:i + 1].astype(np.float32),
            X_train_feat[i:i + 1].astype(np.float32)
        ]


converter_quant.representative_dataset = representative_dataset_gen
tflite_quant_model = converter_quant.convert()

tflite_quant_path = os.path.join(MODEL_OUTPUT_DIR, "rescbar_fusionnet_int8.tflite")
with open(tflite_quant_path, "wb") as f:
    f.write(tflite_quant_model)
print(f"INT8 TFLite model saved: {tflite_quant_path}")

# ---- Model sizes ----
float_model_size_kb = os.path.getsize(tflite_float_path) / 1024.0
quant_model_size_kb = os.path.getsize(tflite_quant_path) / 1024.0
print(f"\nFloat32 model size : {float_model_size_kb:.2f} KB ({float_model_size_kb / 1024:.2f} MB)")
print(f"INT8 model size    : {quant_model_size_kb:.2f} KB ({quant_model_size_kb / 1024:.2f} MB)")
print(f"Size reduction     : {(1 - quant_model_size_kb / float_model_size_kb) * 100:.1f}%")


# ============================================================
# 45. EDGE PERFORMANCE METRICS
# ============================================================

print("\n" + "=" * 70)
print("STEP 18: EDGE PERFORMANCE METRICS")
print("=" * 70)


def estimate_model_flops(model):
    """Estimate total FLOPs for a Keras model by layer type."""
    total_flops = 0
    for layer in model.layers:
        try:
            out_shape = layer.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            in_shape = layer.input_shape
            if isinstance(in_shape, list):
                in_shape = in_shape[0]

            if isinstance(layer, tf.keras.layers.Conv1D):
                k = layer.kernel_size[0]
                c_in = in_shape[-1]
                c_out = layer.filters
                out_len = out_shape[1] if len(out_shape) > 2 else 1
                total_flops += 2 * out_len * k * c_in * c_out

            elif isinstance(layer, tf.keras.layers.Dense):
                in_dim = in_shape[-1]
                out_dim = layer.units
                total_flops += 2 * in_dim * out_dim

            elif isinstance(layer, tf.keras.layers.GRU):
                timesteps = in_shape[1] if len(in_shape) > 2 else 1
                input_dim = in_shape[-1]
                units = layer.units
                # GRU has 3 gates, each with input and recurrent weights
                total_flops += 2 * timesteps * 3 * (input_dim * units + units * units)

            elif isinstance(layer, tf.keras.layers.Bidirectional):
                inner = layer.forward_layer
                if isinstance(inner, tf.keras.layers.GRU):
                    timesteps = in_shape[1] if len(in_shape) > 2 else 1
                    input_dim = in_shape[-1]
                    units = inner.units
                    total_flops += 2 * 2 * timesteps * 3 * (input_dim * units + units * units)
        except Exception:
            continue
    return total_flops


def run_tflite_inference(tflite_model_content, X_seq, X_feat):
    """Run TFLite inference on all test samples. Returns predictions, latencies."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Match input details to sequence vs feature by shape dimensionality
    seq_detail = None
    feat_detail = None
    for d in input_details:
        if len(d["shape"]) == 3:
            seq_detail = d
        else:
            feat_detail = d

    # Resize to single-sample inference
    interpreter.resize_tensor_input(seq_detail["index"], [1] + list(X_seq.shape[1:]))
    interpreter.resize_tensor_input(feat_detail["index"], [1] + list(X_feat.shape[1:]))
    interpreter.allocate_tensors()

    n_samples = len(X_seq)
    predictions = []
    latencies = []

    for i in range(n_samples):
        seq_sample = X_seq[i:i + 1].astype(np.float32)
        feat_sample = X_feat[i:i + 1].astype(np.float32)

        interpreter.set_tensor(seq_detail["index"], seq_sample)
        interpreter.set_tensor(feat_detail["index"], feat_sample)

        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()

        output = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(np.argmax(output, axis=1)[0])
        latencies.append((t1 - t0) * 1000.0)  # ms

    return np.array(predictions), np.array(latencies)


# ---- FLOPs estimation ----
total_flops = estimate_model_flops(final_model)
print(f"Estimated FLOPs : {total_flops:,.0f} ({total_flops / 1e6:.2f} MFLOPs)")

# ---- Float32 TFLite inference ----
print("\nRunning Float32 TFLite inference on test set...")
process = psutil.Process(os.getpid())

gc.collect()
mem_before_float = process.memory_info().rss / (1024 * 1024)  # MB
cpu_before_float = psutil.cpu_percent(interval=None)

tracemalloc.start()
t_start_float = time.time()

float_preds, float_latencies = run_tflite_inference(tflite_float_model, X_test, X_test_feat)

t_end_float = time.time()
float_peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # MB
tracemalloc.stop()

cpu_after_float = psutil.cpu_percent(interval=0.1)
mem_after_float = process.memory_info().rss / (1024 * 1024)  # MB

float_total_time = t_end_float - t_start_float
float_avg_latency = np.mean(float_latencies)
float_throughput = len(X_test) / float_total_time
float_accuracy = accuracy_score(test_true, float_preds)
float_ram_usage = mem_after_float - mem_before_float
float_cpu_util = max(cpu_after_float, cpu_before_float)

# Estimate energy: CPU_time * TDP_estimate (assume 5W for mobile edge device)
TDP_WATTS = 5.0
float_energy_mj = float_total_time * TDP_WATTS * 1000  # millijoules

print(f"Float32 Accuracy      : {float_accuracy:.4f}")
print(f"Float32 Inference Time: {float_total_time:.4f} s")
print(f"Float32 Avg Latency   : {float_avg_latency:.4f} ms")
print(f"Float32 Throughput    : {float_throughput:.2f} samples/sec")

# ---- INT8 TFLite inference ----
print("\nRunning INT8 TFLite inference on test set...")

gc.collect()
mem_before_quant = process.memory_info().rss / (1024 * 1024)
cpu_before_quant = psutil.cpu_percent(interval=None)

tracemalloc.start()
t_start_quant = time.time()

quant_preds, quant_latencies = run_tflite_inference(tflite_quant_model, X_test, X_test_feat)

t_end_quant = time.time()
quant_peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
tracemalloc.stop()

cpu_after_quant = psutil.cpu_percent(interval=0.1)
mem_after_quant = process.memory_info().rss / (1024 * 1024)

quant_total_time = t_end_quant - t_start_quant
quant_avg_latency = np.mean(quant_latencies)
quant_throughput = len(X_test) / quant_total_time
quant_accuracy = accuracy_score(test_true, quant_preds)
quant_ram_usage = mem_after_quant - mem_before_quant
quant_cpu_util = max(cpu_after_quant, cpu_before_quant)
quant_energy_mj = quant_total_time * TDP_WATTS * 1000

print(f"INT8 Accuracy      : {quant_accuracy:.4f}")
print(f"INT8 Inference Time: {quant_total_time:.4f} s")
print(f"INT8 Avg Latency   : {quant_avg_latency:.4f} ms")
print(f"INT8 Throughput    : {quant_throughput:.2f} samples/sec")

# ---- Summary table ----
print("\n" + "=" * 70)
print("EDGE PERFORMANCE METRICS SUMMARY")
print("=" * 70)
print(f"{'Metric':<40s} {'Float32':>14s} {'INT8':>14s}")
print("-" * 70)
print(f"{'Model Size (KB)':<40s} {float_model_size_kb:>14.2f} {quant_model_size_kb:>14.2f}")
print(f"{'Memory Usage (MB)':<40s} {float_peak_mem:>14.2f} {quant_peak_mem:>14.2f}")
print(f"{'Total Inference Time (s)':<40s} {float_total_time:>14.4f} {quant_total_time:>14.4f}")
print(f"{'FLOPs (M)':<40s} {total_flops / 1e6:>14.2f} {total_flops / 1e6:>14.2f}")
print(f"{'Avg Inference Latency (ms)':<40s} {float_avg_latency:>14.4f} {quant_avg_latency:>14.4f}")
print(f"{'Throughput (samples/sec)':<40s} {float_throughput:>14.2f} {quant_throughput:>14.2f}")
print(f"{'Energy Consumption (mJ)':<40s} {float_energy_mj:>14.2f} {quant_energy_mj:>14.2f}")
print(f"{'CPU Utilization (%)':<40s} {float_cpu_util:>14.1f} {quant_cpu_util:>14.1f}")
print(f"{'RAM Usage (MB)':<40s} {abs(float_ram_usage):>14.2f} {abs(quant_ram_usage):>14.2f}")
print(f"{'Accuracy':<40s} {float_accuracy:>14.4f} {quant_accuracy:>14.4f}")

# ============================================================
# 46. PLOT SETUP — GLOBAL STYLING
# ============================================================
# All plots: figsize=(10,8), fontweight='bold', fontfamily='Times New Roman',
# no grid, different colors per plot, values above bars, wave-style lines.

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.figsize"] = (10, 8)

print("\n" + "=" * 70)
print("GENERATING ALL PLOTS (SEPARATE FIGURES)")
print("=" * 70)


def wave_smooth(x_data, y_data, num_points=300):
    """Smooth a curve using cubic spline interpolation for wave-like appearance."""
    x_arr = np.array(x_data, dtype=float)
    y_arr = np.array(y_data, dtype=float)
    if len(x_arr) < 4:
        return x_arr, y_arr
    x_new = np.linspace(x_arr.min(), x_arr.max(), num_points)
    spl = make_interp_spline(x_arr, y_arr, k=3)
    y_new = spl(x_new)
    return x_new, y_new


def add_bar_values(ax, bars, fmt=".4f", fontsize=9):
    """Add value labels above each bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, height,
            f"{height:{fmt}}", ha="center", va="bottom",
            fontweight="bold", fontfamily="Times New Roman", fontsize=fontsize
        )


# ============================================================
# PLOT 1: Training Accuracy vs Validation Accuracy (Wave Plot)
# ============================================================

fig1, ax1 = plt.subplots(num="Training vs Validation Accuracy", figsize=(10, 8))
epochs_range = np.arange(1, len(history.history["accuracy"]) + 1)

x_sm, y_sm_train = wave_smooth(epochs_range, history.history["accuracy"])
x_sm, y_sm_val = wave_smooth(epochs_range, history.history["val_accuracy"])

ax1.plot(x_sm, y_sm_train, color="#E74C3C", linewidth=2.5, label="Training Accuracy")
ax1.plot(x_sm, y_sm_val, color="#2ECC71", linewidth=2.5, label="Validation Accuracy")
ax1.scatter(epochs_range, history.history["accuracy"], color="#E74C3C", s=40, zorder=5)
ax1.scatter(epochs_range, history.history["val_accuracy"], color="#2ECC71", s=40, zorder=5)

for i, (tr, va) in enumerate(zip(history.history["accuracy"], history.history["val_accuracy"])):
    ax1.annotate(f"{tr:.3f}", (epochs_range[i], tr), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=7, fontweight="bold",
                 fontfamily="Times New Roman", color="#E74C3C")
    ax1.annotate(f"{va:.3f}", (epochs_range[i], va), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=7, fontweight="bold",
                 fontfamily="Times New Roman", color="#2ECC71")

ax1.set_title("ResCBAR-FusionNet: Training vs Validation Accuracy",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax1.set_xlabel("Epoch", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax1.set_ylabel("Accuracy", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax1.legend(fontsize=12, prop={"family": "Times New Roman", "weight": "bold"})
ax1.grid(False)
ax1.tick_params(labelsize=11)
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig1.tight_layout()
fig1.savefig(os.path.join(OUTPUT_PLOTS_DIR, "01_accuracy_curve.png"), dpi=300)

# ============================================================
# PLOT 2: Training Loss vs Validation Loss (Wave Plot)
# ============================================================

fig2, ax2 = plt.subplots(num="Training vs Validation Loss", figsize=(10, 8))
x_sm2, y_sm_loss_tr = wave_smooth(epochs_range, history.history["loss"])
x_sm2, y_sm_loss_val = wave_smooth(epochs_range, history.history["val_loss"])

ax2.plot(x_sm2, y_sm_loss_tr, color="#3498DB", linewidth=2.5, label="Training Loss")
ax2.plot(x_sm2, y_sm_loss_val, color="#E67E22", linewidth=2.5, label="Validation Loss")
ax2.scatter(epochs_range, history.history["loss"], color="#3498DB", s=40, zorder=5)
ax2.scatter(epochs_range, history.history["val_loss"], color="#E67E22", s=40, zorder=5)

for i, (tr, va) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
    ax2.annotate(f"{tr:.3f}", (epochs_range[i], tr), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=7, fontweight="bold",
                 fontfamily="Times New Roman", color="#3498DB")
    ax2.annotate(f"{va:.3f}", (epochs_range[i], va), textcoords="offset points",
                 xytext=(0, -15), ha="center", fontsize=7, fontweight="bold",
                 fontfamily="Times New Roman", color="#E67E22")

ax2.set_title("ResCBAR-FusionNet: Training vs Validation Loss",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax2.set_xlabel("Epoch", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax2.set_ylabel("Loss", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax2.legend(fontsize=12, prop={"family": "Times New Roman", "weight": "bold"})
ax2.grid(False)
ax2.tick_params(labelsize=11)
for label in ax2.get_xticklabels() + ax2.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_PLOTS_DIR, "02_loss_curve.png"), dpi=300)

# ============================================================
# PLOT 3: ROC Curve per Class
# ============================================================

fig3, ax3 = plt.subplots(num="ROC Curve per Class", figsize=(10, 8))
roc_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6", "#F39C12"]

for i in range(NUM_CLASSES):
    fpr_roc, tpr_roc, _ = roc_curve(y_test_enc[:, i], test_pred_probs[:, i])
    roc_auc = sklearn_auc(fpr_roc, tpr_roc)
    # Smooth ROC curve
    if len(fpr_roc) > 4:
        fpr_sm, tpr_sm = wave_smooth(fpr_roc, tpr_roc, num_points=200)
        # Clip to valid range
        tpr_sm = np.clip(tpr_sm, 0, 1)
    else:
        fpr_sm, tpr_sm = fpr_roc, tpr_roc
    ax3.plot(fpr_sm, tpr_sm, color=roc_colors[i % len(roc_colors)],
             linewidth=2.5, label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.4f})")

ax3.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.5)
ax3.set_title("ROC Curve per Class",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax3.set_xlabel("False Positive Rate", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax3.set_ylabel("True Positive Rate", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax3.legend(fontsize=11, prop={"family": "Times New Roman", "weight": "bold"}, loc="lower right")
ax3.grid(False)
ax3.tick_params(labelsize=11)
for label in ax3.get_xticklabels() + ax3.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig3.tight_layout()
fig3.savefig(os.path.join(OUTPUT_PLOTS_DIR, "03_roc_curve.png"), dpi=300)

# ============================================================
# PLOT 4: Precision-Recall Curve per Class
# ============================================================

fig4, ax4 = plt.subplots(num="Precision-Recall Curve per Class", figsize=(10, 8))
pr_colors = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD", "#D35400"]

for i in range(NUM_CLASSES):
    prec_curve, rec_curve, _ = precision_recall_curve(y_test_enc[:, i], test_pred_probs[:, i])
    ap = average_precision_score(y_test_enc[:, i], test_pred_probs[:, i])
    # Smooth precision-recall curve
    if len(rec_curve) > 4:
        # Sort by recall for smooth interpolation
        sorted_idx = np.argsort(rec_curve)
        rec_sorted = rec_curve[sorted_idx]
        prec_sorted = prec_curve[sorted_idx]
        rec_sm, prec_sm = wave_smooth(rec_sorted, prec_sorted, num_points=200)
        prec_sm = np.clip(prec_sm, 0, 1)
    else:
        rec_sm, prec_sm = rec_curve, prec_curve
    ax4.plot(rec_sm, prec_sm, color=pr_colors[i % len(pr_colors)],
             linewidth=2.5, label=f"{CLASS_NAMES[i]} (AP={ap:.4f})")

ax4.set_title("Precision-Recall Curve per Class",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax4.set_xlabel("Recall", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax4.set_ylabel("Precision", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax4.legend(fontsize=11, prop={"family": "Times New Roman", "weight": "bold"}, loc="lower left")
ax4.grid(False)
ax4.tick_params(labelsize=11)
for label in ax4.get_xticklabels() + ax4.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig4.tight_layout()
fig4.savefig(os.path.join(OUTPUT_PLOTS_DIR, "04_precision_recall_curve.png"), dpi=300)

# ============================================================
# PLOT 5: FPR and FNR Bar Plot per Class
# ============================================================

fig5, ax5 = plt.subplots(num="FPR and FNR per Class", figsize=(10, 8))
x_pos5 = np.arange(NUM_CLASSES)
bar_w5 = 0.35

bars_fpr = ax5.bar(x_pos5 - bar_w5 / 2, fpr_per_class, width=bar_w5,
                   color="#8E44AD", label="FPR (False Positive Rate)")
bars_fnr = ax5.bar(x_pos5 + bar_w5 / 2, fnr_per_class, width=bar_w5,
                   color="#E74C3C", label="FNR (False Negative Rate)")

add_bar_values(ax5, bars_fpr, fmt=".4f", fontsize=9)
add_bar_values(ax5, bars_fnr, fmt=".4f", fontsize=9)

ax5.set_xticks(x_pos5)
ax5.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
ax5.set_title("FPR and FNR per Class",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax5.set_xlabel("Activity Class", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax5.set_ylabel("Rate", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax5.legend(fontsize=12, prop={"family": "Times New Roman", "weight": "bold"})
ax5.grid(False)
ax5.tick_params(labelsize=11)
for label in ax5.get_xticklabels() + ax5.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig5.tight_layout()
fig5.savefig(os.path.join(OUTPUT_PLOTS_DIR, "05_fpr_fnr_bar.png"), dpi=300)

# ============================================================
# PLOT 6: Overall Performance Metrics Bar Chart
# ============================================================

fig6, ax6 = plt.subplots(num="Overall Performance Metrics", figsize=(10, 8))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]
metric_values = [test_accuracy, test_precision, test_recall, test_f1, overall_specificity]
metric_colors = ["#1ABC9C", "#3498DB", "#9B59B6", "#E67E22", "#E74C3C"]

bars6 = ax6.bar(metric_names, metric_values, color=metric_colors, width=0.6, edgecolor="black", linewidth=1.2)
add_bar_values(ax6, bars6, fmt=".4f", fontsize=11)

ax6.set_ylim(0, 1.15)
ax6.set_title("Overall Performance Metrics (Test Set)",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax6.set_xlabel("Metric", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax6.set_ylabel("Score", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax6.grid(False)
ax6.tick_params(labelsize=12)
for label in ax6.get_xticklabels() + ax6.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig6.tight_layout()
fig6.savefig(os.path.join(OUTPUT_PLOTS_DIR, "06_overall_performance.png"), dpi=300)

# ============================================================
# PLOT 7: Confusion Matrix
# ============================================================

fig7, ax7 = plt.subplots(num="Confusion Matrix (Test Set)", figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={"fontweight": "bold", "fontfamily": "Times New Roman", "fontsize": 13},
            ax=ax7, linewidths=1.5, linecolor="black")
ax7.set_title("Confusion Matrix — Test Set",
              fontsize=16, fontweight="bold", fontfamily="Times New Roman")
ax7.set_xlabel("Predicted Label", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax7.set_ylabel("True Label", fontsize=13, fontweight="bold", fontfamily="Times New Roman")
ax7.tick_params(labelsize=11)
for label in ax7.get_xticklabels() + ax7.get_yticklabels():
    label.set_fontweight("bold")
    label.set_fontfamily("Times New Roman")
fig7.tight_layout()
fig7.savefig(os.path.join(OUTPUT_PLOTS_DIR, "07_confusion_matrix.png"), dpi=300)

# ============================================================
# EDGE PERFORMANCE PLOTS (8–17): Each metric in its own window
# ============================================================

def make_edge_bar_plot(fig_num, title, labels, values, colors, ylabel, fmt, save_name):
    """Helper to create a styled edge performance bar plot."""
    fig, ax = plt.subplots(num=fig_num, figsize=(10, 8))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    add_bar_values(ax, bars, fmt=fmt, fontsize=12)
    ax.set_title(title, fontsize=16, fontweight="bold", fontfamily="Times New Roman")
    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold", fontfamily="Times New Roman")
    ax.grid(False)
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontfamily("Times New Roman")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PLOTS_DIR, save_name), dpi=300)
    return fig


# PLOT 8: Model Size
make_edge_bar_plot(
    "Model Size Comparison",
    "Model Size: Float32 vs INT8 Quantized",
    ["Float32", "INT8 Quantized"],
    [float_model_size_kb, quant_model_size_kb],
    ["#1ABC9C", "#3498DB"], "Model Size (KB)", ".2f",
    "08_model_size.png"
)

# PLOT 9: Memory Usage
make_edge_bar_plot(
    "Memory Usage",
    "Peak Memory Usage During Inference",
    ["Float32", "INT8 Quantized"],
    [float_peak_mem, quant_peak_mem],
    ["#E67E22", "#F39C12"], "Memory Usage (MB)", ".2f",
    "09_memory_usage.png"
)

# PLOT 10: Inference Time
make_edge_bar_plot(
    "Inference Time",
    "Total Inference Time on Test Set",
    ["Float32", "INT8 Quantized"],
    [float_total_time, quant_total_time],
    ["#9B59B6", "#8E44AD"], "Inference Time (seconds)", ".4f",
    "10_inference_time.png"
)

# PLOT 11: Computational Cost (FLOPs)
make_edge_bar_plot(
    "Computational Cost (FLOPs)",
    "Computational Cost (FLOPs)",
    ["Float32", "INT8 Quantized"],
    [total_flops / 1e6, total_flops / 1e6],
    ["#27AE60", "#2ECC71"], "MFLOPs", ".2f",
    "11_flops.png"
)

# PLOT 12: Inference Latency (ms)
make_edge_bar_plot(
    "Inference Latency",
    "Average Inference Latency per Sample",
    ["Float32", "INT8 Quantized"],
    [float_avg_latency, quant_avg_latency],
    ["#E74C3C", "#C0392B"], "Latency (ms)", ".4f",
    "12_inference_latency.png"
)

# PLOT 13: Throughput
make_edge_bar_plot(
    "Throughput",
    "Inference Throughput (Samples per Second)",
    ["Float32", "INT8 Quantized"],
    [float_throughput, quant_throughput],
    ["#3498DB", "#2980B9"], "Throughput (samples/sec)", ".2f",
    "13_throughput.png"
)

# PLOT 14: Energy Consumption
make_edge_bar_plot(
    "Energy Consumption",
    "Estimated Energy Consumption (Edge Device, TDP=5W)",
    ["Float32", "INT8 Quantized"],
    [float_energy_mj, quant_energy_mj],
    ["#F39C12", "#E67E22"], "Energy (milliJoules)", ".2f",
    "14_energy_consumption.png"
)

# PLOT 15: CPU Utilization
make_edge_bar_plot(
    "CPU Utilization",
    "CPU Utilization During Inference",
    ["Float32", "INT8 Quantized"],
    [float_cpu_util, quant_cpu_util],
    ["#1ABC9C", "#16A085"], "CPU Utilization (%)", ".1f",
    "15_cpu_utilization.png"
)

# PLOT 16: RAM Usage
make_edge_bar_plot(
    "RAM Usage",
    "RAM Usage During Inference",
    ["Float32", "INT8 Quantized"],
    [abs(float_ram_usage), abs(quant_ram_usage)],
    ["#D35400", "#E74C3C"], "RAM Usage (MB)", ".2f",
    "16_ram_usage.png"
)

# PLOT 17: Accuracy Before vs After Quantization
make_edge_bar_plot(
    "Accuracy: Before vs After Quantization",
    "Accuracy Before vs After Quantization",
    ["Float32\n(Before)", "INT8 Quantized\n(After)"],
    [float_accuracy, quant_accuracy],
    ["#2ECC71", "#E74C3C"], "Accuracy", ".4f",
    "17_accuracy_quantization.png"
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL PLOTS SAVED TO:", OUTPUT_PLOTS_DIR)
print("=" * 70)
plot_files = [
    "01_accuracy_curve.png",
    "02_loss_curve.png",
    "03_roc_curve.png",
    "04_precision_recall_curve.png",
    "05_fpr_fnr_bar.png",
    "06_overall_performance.png",
    "07_confusion_matrix.png",
    "08_model_size.png",
    "09_memory_usage.png",
    "10_inference_time.png",
    "11_flops.png",
    "12_inference_latency.png",
    "13_throughput.png",
    "14_energy_consumption.png",
    "15_cpu_utilization.png",
    "16_ram_usage.png",
    "17_accuracy_quantization.png",
]
for pf in plot_files:
    print(f" - {pf}")

print("\n" + "=" * 70)
print("MODEL FILES SAVED TO:", MODEL_OUTPUT_DIR)
print("=" * 70)
print(f" - rescbar_fusionnet_final.h5          (Keras)")
print(f" - rescbar_fusionnet_float32.tflite    (TFLite Float32)")
print(f" - rescbar_fusionnet_int8.tflite       (TFLite INT8 Quantized)")

print("\n" + "=" * 70)
print("EDGE DEPLOYMENT COMPLETE")
print("=" * 70)
print(f"Training Epochs     : {FINAL_EPOCHS}")
print(f"Test Accuracy       : {test_accuracy:.4f}")
print(f"Float32 TFLite Acc  : {float_accuracy:.4f}")
print(f"INT8 TFLite Acc     : {quant_accuracy:.4f}")
print(f"Model Size Reduction: {(1 - quant_model_size_kb / float_model_size_kb) * 100:.1f}%")
print(f"Latency Improvement : {(1 - quant_avg_latency / float_avg_latency) * 100:.1f}%")

plt.show()

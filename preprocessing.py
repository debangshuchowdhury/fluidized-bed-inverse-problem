import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy.signal import butter, filtfilt


warnings.filterwarnings("ignore", category=FutureWarning)


"""
    HELPER FUNCTIONS
"""


def read_and_process_data(file_path: str, is_excel: bool = False):
    """
    Reads and processes data from a given file path.
    If the file is an Excel file, it uses the openpyxl engine to read it.
    Converts any string representations of numbers with commas to floats
    Calculates the 'Minutes' column based on the 'Time' column if it exists.

    Args:
        file_path (str): Path to the data file.
        is_excel (bool): Boolean indicating if the file is an Excel file.

    Returns:
        A pandas DataFrame containing only the raw data in numeric format for further
    """

    if is_excel:
        data = pd.read_excel(file_path, engine="openpyxl")
    else:
        data = pd.read_csv(file_path, delimiter="\t")

    # data = data.apply(
    #     lambda x: (
    #         x.str.replace(",", ".").str.strip().str.split(r"\s+|\t").str[0]
    #         if x.dtype == "object"
    #         else x
    #     )
    # )

    numeric_cols = [
        "mfc1",
        "mfc2",
        "mfc3",
        "mfc4",
        "mfc5",
        "mfc6",
        "mfc7",
        "mfc8",
        "mfc9",
        "p1",
        "distance",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = (
                data[col]
                .astype(str)
                .str.replace(",", ".")
                .str.strip()
                .str.split(r"\s+|\t")
                .str[0]
            )
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Time" in data.columns:
        temp1 = pd.to_datetime(data["Time"], format="%H:%M:%S.%f", errors="coerce")
        if temp1.isna().any():
            temp1 = pd.to_datetime(data["Time"], errors="coerce")

        if temp1.notna().all():
            data["Time"] = temp1.copy()
            data["Minutes"] = (
                data["Time"] - data["Time"].iloc[0]
            ).dt.total_seconds() / 60.0
        else:
            data["Minutes"] = (data["Time"] - data["Time"].iloc[0]) / (60 * 1e9)

    # data = data.apply(pd.to_numeric, errors="coerce")
    return data


def calculate_metrics(
    Data: pd.DataFrame, filled_height: float, material: str, fl_step: int
):
    """
    Computes total bed height, bed expansion, normalized pressure, and flow rates for Level 1 and Level 2.
    The flow rates are calculated differently based on the material type (alumina or sand).

    Args:
        Data (pd.DataFrame): The raw data DataFrame retrieved from the read_and_process_data method.
        filled_height (float): The initial bed height in meters.
        material (str): The type of material ('alumina' or 'sand').
        fl_step (int): The step size for flow rate.

    Returns:
        A pandas DataFrame with the calculated metrics added as new columns.
    """

    TOTAL_H = 4.313
    data = Data.copy()
    data["total_bed_height"] = TOTAL_H - data["distance"]
    data["bed_exp"] = TOTAL_H - filled_height - data["distance"]
    data["normalized_p"] = data["p1"] / filled_height

    if material == "alumina":
        data["fl_L1"] = data["mfc1"] + data["mfc2"] + data["mfc3"]
        data["fl_L2"] = data["mfc4"] + data["mfc5"] + data["mfc6"]

        if data["fl_L1"].mean() < 2 * fl_step:
            print("choosing mfc1,2,3 was incorrect. taking 789 instead.")
            data["fl_L1"] = data["mfc7"] + data["mfc8"] + data["mfc9"]

    elif material == "sand":
        data["fl_L1"] = data["mfc7"] + data["mfc8"] + data["mfc9"]
        data["fl_L2"] = data["mfc4"] + data["mfc5"] + data["mfc6"]

        if data["fl_L1"].mean() < 4 * fl_step:
            print("choosing mfc789 was incorrect. taking 123 instead.")
            data["fl_L1"] = data["mfc1"] + data["mfc2"] + data["mfc3"]
    else:
        raise ValueError("material must be either sand or alumina")

    data["flowrate_combined"] = data["fl_L1"] + data["fl_L2"]

    return data


def butter_filter(data: pd.DataFrame, frequency: float):
    """
    Applies a Butterworth low-pass filter to the metrics.

    Args:
        data (pd.DataFrame): The input DataFrame containing the calculated data and metrics.
        frequency (float): The cutoff frequency for the Butterworth filter.

    Returns:
        A pandas DataFrame with the filtered metrics.
    """

    # Filter requirements.
    fs = frequency
    cutoff = 2  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2
    normal_cutoff = cutoff / nyq

    # Filter
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def filter(x: pd.DataFrame, f: float):
    """
    Applies the Butterworth filter to the each column of the input DataFrame.

    Args:
        x (pd.DataFrame): The input DataFrame containing the calculated data and metrics.
        f (float): The cutoff frequency for the Butterworth filter.

    Returns:
        A pandas DataFrame with the filtered metrics.
    """

    for k in range(1, x.shape[-1]):
        if pd.api.types.is_numeric_dtype(x.iloc[0, k]):
            x.iloc[:, k] = butter_filter(x.iloc[:, k], f)

    return x


def recover_averaged_data(
    data: pd.DataFrame,
    freq: float,
    step_size_fl: float,
    step_duration: float,
    features: list,
    initialbed: float,
):
    """
    Identify steady-state regions in flow rate data and compute
    averaged feature values for each steady segment.
    The number of points discarded after a step change can be controlled by adjusting the window size and threshold parameters.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing at least the column 'flowrate_combined'
        and the specified feature columns.
    freq : float or int
        Sampling frequency of the data (Hz).
    step_size_fl : float or int
        Smallest flow rate step size used in the experiment to define steady-state
        tolerance.
    step_duration : float
        Duration of each step (seconds).
    features : list of str
        List of column names to average over steady-state regions.
    initialbed : float or int
        Initial bed height value to append to the resulting dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing mean values of the specified features
        for each detected steady-state segment, including the
        initial bed height.
    """

    # window that slides across the data to find the averages, set to 20% of the step duration
    window = int(step_duration * 0.2 * freq)

    # Calculates the rolling difference within the window
    difing = (
        data["flowrate_combined"].rolling(window).max()
        - data["flowrate_combined"].rolling(window).min()
    )

    # Thresholds the rolling difference to identify indices where the flow rate is considered steady
    threshold = int(step_size_fl) * 0.2
    inds = np.where(np.abs(difing) <= threshold)[0]

    # Extracts the relevant features for the identified steady indices and adds the initial bed height as a feature
    steady_data = pd.DataFrame(data.iloc[inds][features])
    steady_data["initial_bed_height"] = initialbed

    steady_data = steady_data.dropna(subset=features + ["initial_bed_height"])

    # Identifies a jump in the data after the rolling mean
    jumps = np.where(np.diff(inds) > 1)[0]
    starts = np.zeros(len(jumps) + 1, dtype=int)
    ends = np.zeros_like(starts)
    ends[-1] = data.shape[0] - 1

    for i, jump in enumerate(jumps):
        starts[i + 1] = jump + 1
        ends[i] = jump

    steady_means = pd.DataFrame(columns=steady_data.columns)

    for start, end in zip(starts, ends):
        mean_row = steady_data.iloc[start : end + 1].mean(axis=0).to_frame().T
        steady_means = pd.concat(
            [steady_means, mean_row],
            ignore_index=True,
        )

    return steady_means


"""
    USER INPUTS
"""
# Path to the folder containing the data files
file_path = "good_data/sand"
# Material type (e.g., 'alumina' or 'sand')
material = "sand"
# To save the processed data
to_save = False
# To split into forwards and backwards runs
to_split = True
# To plot the processed data for each experiment
to_plot = True
# List of relevant features to be used in the final dataset
relevant_features = [
    "Minutes",
    "fl_L1",
    "fl_L2",
    "total_bed_height",
    "p1",
    "flowrate_combined",
    "bed_exp",
    "normalized_p",
]


"""
    PROCESSING SCRIPT
"""
folder = Path(file_path)
# The exp file names must be in the format: initialbedheight_flowratestepsize_stepduration_frequency_runtype
FINAL = pd.DataFrame(columns=relevant_features)
FINAL_backwards = pd.DataFrame(columns=relevant_features)

for file in folder.iterdir():
    print(file.name)
    if not file.is_file():
        continue

    initial_bed_height, step_size_fl, step_duration, frequency, run_type = (
        file.stem.split("_")
    )
    initial_bed_height = float(initial_bed_height) / 1000
    frequency = int(frequency)
    step_size_fl = int(step_size_fl)
    step_duration = int(step_duration)

    if step_duration < 100:
        print(
            f"Step duration too small for steady state assumption. Skipping {file.name}."
        )
        continue

    # if skip:
    #     if (
    #         initial_bed_height == 0.956
    #         # initial_bed_height == 0.889
    #         # or initial_bed_height == 0.808
    #         or initial_bed_height == 0.903
    #         or initial_bed_height == 0.872
    #         # or initial_bed_height == 0.897
    #     ):
    #         print(f"skipping {initial_bed_height} for testing")
    #         continue

    if file.suffix == ".xlsx":
        isexcel = True
    else:
        isexcel = False

    data = read_and_process_data(file_path + "/" + file.name, isexcel)
    # print(data.isna().any())
    # nan_rows = data[data["p1"].isna()]
    # print("p1 nan rows = ", nan_rows["p1"])
    # print("dtypes")
    # print(data.dtypes)
    metrics = filter(
        calculate_metrics(data, initial_bed_height, material, step_size_fl),
        frequency,
    )

    # if to_split:
    #     run_type = "forward"
    # print("metrics", metrics.isna().any())
    # print(metrics.dtypes)
    # print("flowrate combined dtype = ", metrics["flowrate_combined"].head())

    segments = recover_averaged_data(
        metrics,
        frequency,
        step_size_fl,
        step_duration,
        relevant_features,
        initial_bed_height,
    )

    if to_plot:
        fig, p = plt.subplots(1, 1)
        p.plot(metrics["Minutes"], metrics["flowrate_combined"], label="Total flowrate")
        p.plot(metrics["Minutes"], metrics["fl_L1"], label="L1", color="orange")
        p.plot(metrics["Minutes"], metrics["fl_L2"], label="L2", color="green")
        fig.set_figwidth(10)
        fig.set_figheight(7.5)
        p.scatter(
            segments.loc[:, "Minutes"],
            segments.loc[:, "fl_L1"],
            s=20,
            zorder=3,
            c="orange",
        )
        p.scatter(
            segments.loc[:, "Minutes"],
            segments.loc[:, "fl_L2"],
            s=10,
            zorder=3,
            c="green",
        )
        p.set_title(f"{initial_bed_height}m")
        # plt.show()

    print(type(segments))
    if np.any(pd.isna(segments)):
        nancols = segments.isna().any()
        print(nancols)
        raise ValueError("nan in segments")

    segments = segments.dropna(subset=relevant_features)

    max_ind = segments["flowrate_combined"].idxmax()
    print(f"{file.name} included.")

    if run_type == "hysteresis":
        FINAL = pd.concat([FINAL, segments.loc[: max_ind + 1]])
        FINAL["step_size"] = step_size_fl
        FINAL["normalized_p"] = FINAL["p1"] / initial_bed_height
        FINAL_backwards = pd.concat([FINAL_backwards, segments.loc[max_ind:]])
        FINAL_backwards["step_size"] = step_size_fl
        FINAL_backwards["normalized_p"] = FINAL_backwards["p1"] / initial_bed_height
        if to_plot:
            p.scatter(
                segments.loc[: max_ind + 1, "Minutes"],
                segments.loc[: max_ind + 1, "flowrate_combined"],
                c="blue",
                s=10,
                zorder=3,
                label="forward run",
            )
            p.scatter(
                segments.loc[max_ind:, "Minutes"],
                segments.loc[max_ind:, "flowrate_combined"],
                c="red",
                s=10,
                zorder=3,
                label="backward run",
            )
            p.legend()
            plt.show()
    else:
        FINAL = pd.concat([FINAL, segments])
        FINAL["step_size"] = step_size_fl
        FINAL["normalized_p"] = FINAL["p1"] / initial_bed_height
        if to_plot:
            p.scatter(
                segments["Minutes"],
                segments["flowrate_combined"],
                c="black",
                s=10,
                zorder=3,
            )
            p.legend()
            plt.show()

if to_save:
    if material == "sand":
        if to_split:
            FINAL.to_csv("sand_data_total.csv", index=False)
        else:
            FINAL.to_csv("sand_data.csv", index=False)
            FINAL_backwards.to_csv("sand_data_backwards.csv", index=False)

    elif material == "alumina":
        if to_split:
            FINAL.to_csv("alumina_data_total.csv", index=False)
        else:
            FINAL.to_csv("alumina_data.csv", index=False)
            FINAL_backwards.to_csv("alumina_data_backwards.csv", index=False)
    else:
        raise ValueError("incorrect material input")
print("final shape = ", FINAL.shape)
print("final backwards shape = ", FINAL_backwards.shape)

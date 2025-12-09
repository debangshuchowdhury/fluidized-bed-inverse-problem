import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def read_and_process_data(file_path, isexcel=False):
    if isexcel:
        data = pd.read_excel(file_path, engine="openpyxl")
    else:
        data = pd.read_csv(file_path, delimiter="\t")
    # data = pd.read_csv(file_path, delimiter="\t")

    data = data.apply(lambda x: x.str.replace(",", ".") if x.dtype == "object" else x)

    if "Time" in data.columns:
        temp1 = pd.to_datetime(data["Time"], format="%H:%M:%S.%f", errors="coerce")

        if np.any(np.isnan(temp1)):
            data["Minutes"] = (data["Time"] - data["Time"].iloc[0]) / (60 * 1e9)
        else:
            data["Time"] = temp1.copy()
            data["Minutes"] = (
                data["Time"] - data["Time"].iloc[0]
            ).dt.total_seconds() / 60.0

    data = data.apply(pd.to_numeric, errors="coerce")
    return data


def calculate_metrics(Data, filled_height, material, fl_step):
    TOTAL_H = 4.313
    data = Data.copy()
    data["total_bed_height"] = TOTAL_H - data["distance"]
    data["bed_exp"] = TOTAL_H - filled_height - data["distance"]

    if material == "alumina":
        data["fl_L1"] = data["mfc1"] + data["mfc2"] + data["mfc3"]
        data["fl_L2"] = data["mfc4"] + data["mfc5"] + data["mfc6"]

        if data["fl_L1"].mean() < 4 * fl_step:
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

    data["flowrate_combi"] = data["fl_L1"] + data["fl_L2"]
    data["total_flowrate"] = (
        data["mfc1"]
        + data["mfc2"]
        + data["mfc3"]
        + data["mfc4"]
        + data["mfc5"]
        + data["mfc6"]
        + data["mfc7"]
        + data["mfc8"]
        + data["mfc9"]
    )

    return data


def butter_filter(data, frequency):
    # Filter requirements.
    fs = frequency
    cutoff = 2  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2

    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


def filter(x, f):
    for k in range(x.shape[-1]):
        x.iloc[:, k] = butter_filter(x.iloc[:, k], f)
    return x


def recover_averaged_data(
    data, freq, step_size_fl, step_duration, features, initialbed
):
    difing = data["total_flowrate"].rolling(
        int(step_duration * 0.1 * freq)
    ).max() - data["total_flowrate"].rolling(int(step_duration * 0.1 * freq)).min(
        step_duration * freq
    )

    inds = np.where(np.abs(difing) <= int(step_size_fl) * 0.2)[0]

    steady_data = pd.DataFrame(data.iloc[inds][features])
    steady_data["initial_bed_height"] = initialbed

    jumps = np.where(np.diff(inds) > 1)[0]
    starts = np.zeros(len(jumps) + 1, int)
    ends = np.zeros_like(starts, int)
    ends[-1] = data.shape[0] - 1

    for i, jump in zip(range(len(jumps)), jumps):
        starts[i + 1] = jump + 1
        ends[i] = jump

    return [
        steady_data.iloc[s : e + 1].mean(axis=0).to_frame().T
        for s, e in zip(starts, ends)
    ]


def recover_averaged_data_array(
    data, freq, step_size_fl, step_duration, features, initialbed
):
    difing = data["flowrate_combi"].rolling(
        int(step_duration * 0.1 * freq)
    ).max() - data["flowrate_combi"].rolling(int(step_duration * 0.1 * freq)).min(
        step_duration * freq
    )

    inds = np.where(np.abs(difing) <= int(step_size_fl) * 0.2)[0]

    steady_data = pd.DataFrame(data.iloc[inds][features])
    steady_data["initial_bed_height"] = initialbed

    jumps = np.where(np.diff(inds) > 1)[0]
    starts = np.zeros(len(jumps) + 1, int)
    ends = np.zeros_like(starts, int)
    ends[-1] = data.shape[0] - 1

    for i, jump in zip(range(len(jumps)), jumps):
        starts[i + 1] = jump + 1
        ends[i] = jump

    steady_means = pd.DataFrame(columns=steady_data.columns)
    for s, e in zip(starts, ends):
        steady_means = pd.concat(
            [steady_means, steady_data.iloc[s : e + 1].mean(axis=0).to_frame().T],
            ignore_index=True,
        )
    return steady_means

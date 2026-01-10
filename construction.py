import numpy as np
import pandas as pd
import helpers.processing as processing
import matplotlib.pyplot as plt
from pathlib import Path


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


file_path = "good_data/alumina"
mater = "alumina"
folder = Path(file_path)
relevant_features = [
    "Minutes",
    "total_flowrate",
    "fl_L1",
    "fl_L2",
    "total_bed_height",
    "p1",
    "flowrate_combi",
    "bed_exp",
]

to_plot = True
to_save = False
skip = False
total = False

FINAL = pd.DataFrame(columns=relevant_features)
FINAL_backwards = pd.DataFrame(columns=relevant_features)

print("Final shape = ", FINAL.shape)

for file in folder.iterdir():
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

    if skip:
        if (
            # initial_bed_height == 0.956
            # initial_bed_height == 0.889
            initial_bed_height == 0.808
            or initial_bed_height == 0.903
            or initial_bed_height == 0.872
            or initial_bed_height == 0.897
        ):
            print(f"skipping {initial_bed_height} for testing")
            continue

    if file.suffix == ".xlsx":
        isexcel = True
    else:
        isexcel = False

    if run_type == "hysteresis":
        data = processing.read_and_process_data(file_path + "/" + file.name, isexcel)
    elif run_type == "forward":
        data = processing.read_and_process_data(file_path + "/" + file.name, isexcel)
    else:
        raise ValueError("the file type is neither .xlsx nor .csv")

    metrics = processing.filter(
        processing.calculate_metrics(data, initial_bed_height, mater, step_size_fl),
        frequency,
    )

    if total:
        run_type = "forward"

    segments = processing.recover_averaged_data_array(
        metrics,
        frequency,
        step_size_fl,
        step_duration,
        relevant_features,
        initial_bed_height,
    )

    if to_plot:
        fig, p = plt.subplots(1, 1)
        p.plot(metrics["Minutes"], metrics["flowrate_combi"], label="Total flowrate")
        p.plot(metrics["Minutes"], metrics["fl_L1"], label="L1", color="orange")
        p.plot(metrics["Minutes"], metrics["fl_L2"], label="L2", color="green")
        # p.plot(metrics["Minutes"], metrics["fl_L3"], label="L3", color="purple")
        # p.plot(
        #     metrics["Minutes"],
        #     metrics["total_flowrate"],
        #     label="All MFCs",
        #     color="red",
        # )
        # p.plot(metrics["Minutes"], metrics["flowrate_combi"], label="specific MFCs")
        fig.set_figwidth(10)
        fig.set_figheight(7.5)
        # p.set_xlim(0, 21)
        # p.set_ylim(0, 1700)
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

    # if segments.shape[0] > 120:
    #     print("Too many datapoints, steady state assumption voided. Skipping file.")
    #     continue

    if np.any(np.isnan(segments)):
        raise ValueError("nan in segments")

    segments = segments.dropna(subset=relevant_features)

    max_ind = segments["flowrate_combi"].idxmax()
    print(f"{file.name} included.")

    if run_type == "hysteresis":
        FINAL = pd.concat([FINAL, segments.loc[: max_ind + 1]])
        FINAL_backwards = pd.concat([FINAL_backwards, segments.loc[max_ind:]])
        if to_plot:
            p.scatter(
                segments.loc[: max_ind + 1, "Minutes"],
                segments.loc[: max_ind + 1, "flowrate_combi"],
                c="blue",
                s=10,
                zorder=3,
                label="forward run",
            )
            p.scatter(
                segments.loc[max_ind:, "Minutes"],
                segments.loc[max_ind:, "flowrate_combi"],
                c="red",
                s=10,
                zorder=3,
                label="backward run",
            )
            p.legend()
            plt.show()
    else:
        FINAL = pd.concat([FINAL, segments])
        if to_plot:
            p.scatter(
                segments["Minutes"],
                segments["flowrate_combi"],
                c="black",
                s=10,
                zorder=3,
            )
            p.legend()
            plt.show()

    # segments = processing.recover_averaged_data(
    #     metrics,
    #     frequency,
    #     step_size_fl,
    #     step_duration,
    #     relevant_features,
    #     initial_bed_height,
    # )

    # # print(f"1. Final shape = {FINAL.shape}")
    # print("segments = ", len(segments))
    # if len(segments) < 100:
    #     for seg in segments:
    #         if to_plot:
    #             p.scatter(
    #                 seg["Minutes"], seg["total_flowrate"], c="black", s=10, zorder=3
    #             )
    #         # print("seg = ", seg)
    #         if np.any(np.isnan(seg)):
    #             raise ValueError("nan in seg")

    #         FINAL = pd.concat([FINAL, seg])
    #     if to_plot:
    #         p.legend()
    #         plt.show()

    # else:
    #     print("too many datapoints, steady state separation unsuccessful. skipping.")


if to_save:
    if total:
        FINAL.to_csv("sand_data_total.csv", index=False)
    else:
        FINAL.to_csv("sand_data_alumina.csv", index=False)
        FINAL_backwards.to_csv("sand_data_alumina_backwards.csv", index=False)
print("final shape = ", FINAL.shape)
print("final backwards shape = ", FINAL_backwards.shape)

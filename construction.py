import numpy as np
import pandas as pd
import helpers.processing as processing
import matplotlib.pyplot as plt
from pathlib import Path


file_path = "good_data/sand"
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
mater = "sand"

to_plot = True


FINAL = pd.DataFrame(columns=relevant_features)
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

    # if run_type == "hysteresis":  # including only the forward runs for now
    #     continue

    if step_duration < 30:
        print("Step duration too small for steady state assumption. Skipping file.")
        continue

    # if not initial_bed_height == 0.681:
    #     continue

    print(f"File name: {file.name}")

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

    if initial_bed_height == 0.956:
        print("skipping 0.813 for testing")
        continue

    metrics = processing.filter(
        processing.calculate_metrics(data, initial_bed_height, mater, step_size_fl),
        frequency,
    )

    if to_plot:
        fig, p = plt.subplots(1, 1)
        p.plot(metrics["Minutes"], metrics["total_flowrate"], label="all MFCs")
        # p.plot(metrics["Minutes"], metrics["flowrate_combi"], label="specific MFCs")
        fig.set_figwidth(10)
        fig.set_figheight(7.5)
        # p.set_xlim(0, 21)
        # p.set_ylim(0, 1700)

    segments = processing.recover_averaged_data(
        metrics,
        frequency,
        step_size_fl,
        step_duration,
        relevant_features,
        initial_bed_height,
    )

    # print(f"1. Final shape = {FINAL.shape}")
    print("segments = ", len(segments))
    if len(segments) < 100:
        for seg in segments:
            if to_plot:
                p.scatter(
                    seg["Minutes"], seg["total_flowrate"], c="black", s=10, zorder=3
                )
            # print("seg = ", seg)
            if np.any(np.isnan(seg)):
                raise ValueError("nan in seg")

            FINAL = pd.concat([FINAL, seg])
        if to_plot:
            p.legend()
            plt.show()

    else:
        print("too many datapoints, steady state separation unsuccessful. skipping.")


# FINAL.to_csv("sand_data.csv", index=False)
print("final shape = ", FINAL.shape)
print("Final columns = ", FINAL.columns)

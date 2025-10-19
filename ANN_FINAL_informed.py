import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import os

start = time.time()
np.random.seed(42)
tf.random.set_seed(42)

### Load dataset
file_path = "CSV_FILES/Test256.xlsx"  # Change to your file path
df = pd.read_excel(file_path)
required_columns = [
    "Mass_Flow_kg_per_s",
    "Particle_Diameter_m",
    "Pressure_Drop_Pa",
    "Bed_Expansion_m",
]
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Excel file must contain columns: {required_columns}")

X = df[["Mass_Flow_kg_per_s", "Particle_Diameter_m"]].values
y = df[["Pressure_Drop_Pa", "Bed_Expansion_m"]].values

### Normalize inputs
mean_X = np.mean(X, axis=0)
std_X = np.std(X, axis=0)
X_scaled = (X - mean_X) / std_X

### Normalize outputs
mean_y = np.mean(y, axis=0)
std_y = np.std(y, axis=0)
y_scaled = (y - mean_y) / std_y

### Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

### Weighted MSE loss based on output variance
output_var = np.var(y_train, axis=0)
weights = np.array(
    [0.45, 0.5, 0.0001]
)  # S1 / output_var  # [correlation, bed height, pressure]

# Input the density from the user
density = float(input("input the density"))  # 2730 from the excel sheet


def pb_error(y_true, y_pred):

    bheight_pred = y_pred[:, 1] * std_y[1] + mean_y[1]
    e = 0.7166
    correlation = (1 - e) * (density - 1.165) * 9.81 * bheight_pred
    pred = (correlation - mean_y[0]) / std_y[0]
    loss_pressure = tf.reduce_mean(tf.abs(y_true[:, 0] - pred))
    loss_height = tf.reduce_mean(tf.abs(y_true[:, 1] - y_pred[:, 1]))
    loss_extra = tf.reduce_mean(tf.abs(y_true[:, 0] - y_pred[:, 0]))
    return (
        weights[0] * loss_pressure + weights[1] * loss_height + loss_extra * weights[2]
    )


### ANN Model
model = Sequential(
    [
        Dense(64, input_shape=(2,)),
        LeakyReLU(alpha=0.01),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dense(64),
        LeakyReLU(alpha=0.01),
        Dense(2),
    ]
)
model.compile(optimizer=Adam(learning_rate=0.001), loss=pb_error, metrics=["mae"])

early_stop = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
history = model.fit(
    X_train, y_train, epochs=5000, validation_split=0, callbacks=[early_stop], verbose=0
)

### Evaluation
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTotal MSE : {loss:.2f} | Total MAE : {mae:.2f}")

# Predictions and denormalization
y_pred_scaled = model.predict(X_test)
y_pred = y_pred_scaled * std_y + mean_y  # denormalize
y_test_real = y_test * std_y + mean_y

y_pred[:, 1] = np.clip(y_pred[:, 1], 0, None)

### Ask user for plot choice
print("Do you want to display curves for:")
print("1 - A specific particle diameter")
print("2 - Multiple diameters in a range")
print("3 - A custom list of particle diameters")
choice = input("Enter your choice (1, 2, or 3): ")

mass_flow_range = np.linspace(
    df["Mass_Flow_kg_per_s"].min(), df["Mass_Flow_kg_per_s"].max(), 50
)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # nouvelle figure

# To collect results for saving
results_data = []


def predict_and_plot(dp_values_m, labels):
    local_data = []
    for dp_val, label in zip(dp_values_m, labels):
        X_plot = np.column_stack(
            [mass_flow_range, np.full_like(mass_flow_range, dp_val)]
        )
        X_plot_scaled = (X_plot - mean_X) / std_X
        y_plot_scaled = model.predict(X_plot_scaled)
        y_plot = y_plot_scaled * std_y + mean_y
        y_plot[:, 1] = np.clip(y_plot[:, 1], 0, None)

        ax[0].plot(mass_flow_range, y_plot[:, 0], "-o", label=label)
        ax[1].plot(mass_flow_range, y_plot[:, 1], "-o", label=label)

        df_res = pd.DataFrame(
            {
                "Mass_Flow_kg_per_s": mass_flow_range,
                "Particle_Diameter_m": dp_val,
                "Pressure_Drop_Pa": y_plot[:, 0],
                "Bed_Expansion_m": y_plot[:, 1],
            }
        )
        local_data.append(df_res)
    return local_data


# def predict_and_plot(dp_values_m, labels):
#     local_data = []

#     # Determine the original max flow rate
#     max_flow_original = np.max(mass_flow_range)

#     # Create extended mass flow range (up to 1.5 times max)
#     extended_max = 1.5 * max_flow_original
#     n_original = len(mass_flow_range)
#     n_extended = int(n_original * 1.5)  # Add 50% more points for extension

#     # Create extended range
#     extended_range = np.linspace(max_flow_original, extended_max, n_extended + 1)[1:] # Exclude first point to avoid dup5lication
#     full_mass_flow_range = np.concatenate([mass_flow_range, extended_range])

#     for dp_val, label in zip(dp_values_m, labels):
#         # Predict for full range (original + extended)
#         # X_plot = np.column_stack([full_mass_flow_range, np.full_like(full_mass_flow_range, dp_val)])
#         X_plot = np.column_stack([full_mass_flow_range, np.full_like(full_mass_flow_range, dp_val)])
#         X_plot_scaled = (X_plot - mean_X) / std_X
#         y_plot_scaled = model.predict(X_plot_scaled)
#         y_plot = y_plot_scaled * std_y + mean_y
#         y_plot[:, 1] = np.clip(y_plot[:, 1], 0, None)

#         # Split predictions into original and extended regions
#         y_original = y_plot[:n_original]
#         y_extended = y_plot[n_original:]

#         # Plot original data
#         ax[0].plot(mass_flow_range, y_original[:, 0], '-o', label=label)
#         ax[1].plot(mass_flow_range, y_original[:, 1], '-o', label=label)

#         # Plot extended data with different color (lighter/dashed style)
#         # Get the color of the last plot to match but make it lighter
#         color = ax[0].lines[-1].get_color()
#         ax[0].plot(extended_range, y_extended[:, 0], '--', color=color, alpha=0.6, linewidth=1.5)
#         ax[1].plot(extended_range, y_extended[:, 1], '--', color=color, alpha=0.6, linewidth=1.5)

#         # Create DataFrame with full range
#         df_res = pd.DataFrame({
#             "Mass_Flow_kg_per_s": full_mass_flow_range,
#             "Particle_Diameter_m": dp_val,
#             "Pressure_Drop_Pa": y_plot[:, 0],
#             "Bed_Expansion_m": y_plot[:, 1],
#             "Region": ["Original"] * n_original + ["Extended"] * n_extended
#         })
#         local_data.append(df_res)

#     return local_data

if choice == "1":
    dp_val = float(input("Enter the particle diameter in microns (e.g. 300): ")) * 1e-6
    labels = [f"{dp_val * 1e6:.0f} µm"]
    results_data += predict_and_plot([dp_val], labels)

elif choice == "2":
    n_diameters_to_plot = int(input("Enter number of curves to plot: "))
    dp_values_m = np.linspace(
        df["Particle_Diameter_m"].min(),
        df["Particle_Diameter_m"].max(),
        n_diameters_to_plot,
    )
    labels = [f"{dp * 1e6:.0f} µm" for dp in dp_values_m]
    results_data += predict_and_plot(dp_values_m, labels)

elif choice == "3":
    dp_list = input("Enter particle diameters in microns separated by commas: ")
    dp_values_m = np.array([float(dp_str) * 1e-6 for dp_str in dp_list.split(",")])
    labels = [f"{dp * 1e6:.0f} µm" for dp in dp_values_m]
    results_data += predict_and_plot(dp_values_m, labels)

else:
    print("Invalid choice. Exiting...")

# Graph formatting
for axis in ax:
    axis.tick_params(axis="both", labelsize=15)  # Ticks formatting

ax[0].set_title("Pressure Drop vs Mass Flow Rate", fontsize=16)
ax[0].set_xlabel("Mass Flow Rate (kg/s)", fontsize=15)
ax[0].set_ylabel("Pressure Drop (Pa)", fontsize=15)
ax[0].grid(True)
ax[0].legend(title="Particle Diameter", fontsize=12, title_fontsize=13)

ax[1].set_title("Bed Expansion vs Mass Flow Rate", fontsize=16)
ax[1].set_xlabel("Mass Flow Rate (kg/s)", fontsize=15)
ax[1].set_ylabel("Bed Expansion Height (m)", fontsize=15)
ax[1].grid(True)
ax[1].legend(title="Particle Diameter", fontsize=12, title_fontsize=13)

plt.tight_layout()


### Ask to save data
folder_name = None
if results_data:
    save_choice = input("Do you want to save the results? (y/n): ").lower()
    if save_choice == "y":
        folder_name = input("Enter folder name for saving results: ")
        os.makedirs(folder_name, exist_ok=True)
        results_df = pd.concat(results_data, ignore_index=True)
        results_df.to_csv(
            os.path.join(folder_name, "simulation_results.csv"), index=False, sep=";"
        )  # Change sep to ":" if values in the same cell
        fig.savefig(os.path.join(folder_name, "simulation_plot.png"), dpi=300)


plt.show()

### Learning curve
fig_lc, ax_lc = plt.subplots()
ax_lc.plot(np.log(history.history["loss"]), label="Training loss")
ax_lc.plot(np.log(history.history["val_loss"]), label="Validation loss")

ax_lc.set_xlabel("Epoch", fontsize=15)
ax_lc.set_ylabel("log(MSE)", fontsize=15)
ax_lc.set_title("Learning Curve", fontsize=16)
ax_lc.legend(fontsize=12)
ax_lc.grid(True)
ax_lc.tick_params(axis="both", labelsize=15)

# Saving learning curve PNG et CSV
if folder_name:
    fig_lc.savefig(os.path.join(folder_name, "learning_curve.png"), dpi=300)
    # CSV for learning curve
    lc_df = pd.DataFrame(
        {
            "Epoch": np.arange(1, len(history.history["loss"]) + 1),
            "Train_loss": history.history["loss"],
            "Validation_loss": history.history["val_loss"],
        }
    )
    lc_df.to_csv(
        os.path.join(folder_name, "learning_curve.csv"), index=False, sep=";"
    )  # Change sep to ":" if values in the same cell
    print(f"Simulation results and learning curve saved in '{folder_name}/' folder.")

plt.show()

### Relative deviation metrics
dev_p = np.abs((y_test_real[:, 0] - y_pred[:, 0]) / y_test_real[:, 0])
dev_h = np.abs((y_test_real[:, 1] - y_pred[:, 1]) / y_test_real[:, 1])
print(f"Average relative deviation for pressure drop (%) : {np.mean(dev_p) * 100:.2f}")
print(
    f"Average relative deviation for bed expansion height (%) : {np.mean(dev_h) * 100:.2f}"
)

end = time.time()
print(f"Execution time: {end - start:.2f} seconds")

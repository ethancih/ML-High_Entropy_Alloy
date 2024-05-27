import numpy as np
import pandas as pd

LOG_BASE = 1.05 #   log base for a couple of terms

# Define the ranges for each variable
# variable_ranges = {"Al": (15, 47),  "Co": (5, 22),
#                    "Cr": (6, 34),   "Cu": (5, 16),
#                    "Fe": (5, 31),   "Ni": (5, 22)   }
variable_ranges = {"Al": (15, 47),  "Co": (0, 22),
                   "Cr": (0, 34),   "Cu": (0, 16),
                   "Fe": (0, 31),   "Ni": (0, 22)   }

# Generate table of data with every possible combination of values
def generate_all_possible_values():
    values_list = []

    min_values = [min_val for min_val, _ in variable_ranges.values()]
    max_values = [max_val for _, max_val in variable_ranges.values()]

    for Al in range(min_values[0], max_values[0] + 1):
        for Co in range(min_values[1], max_values[1] + 1):
            for Cr in range(min_values[2], max_values[2] + 1):
                for Cu in range(min_values[3], max_values[3] + 1):
                    for Fe in range(min_values[4], max_values[4] + 1):
                        for Ni in range(min_values[5], max_values[5] + 1):
                            total = Al + Co + Cr + Cu + Fe + Ni
                            if total == 100:  # 100% when using 1 step increments
                                values_list.append(
                                    {"Al": Al / 10, "Co": Co / 10, "Cr": Cr / 10, "Cu": Cu / 10, "Fe": Fe / 10, "Ni": Ni / 10})
    return values_list


# Generate all possible values
all_possible_values = generate_all_possible_values()

# Multiply all values by 10
for item in all_possible_values:
    for key in item:
        item[key] *= 10

# Determine the number of splits for the data
num_splits = 10
split_size = len(all_possible_values) // num_splits

# Split the data into the specified number of sets
data_sets = [all_possible_values[i * split_size: (i + 1) * split_size] for i in range(num_splits)]

# Convert each set of data into a pandas DataFrame and save it to an Excel file
for i, data_set in enumerate(data_sets):
    df = pd.DataFrame(data_set)
    df.to_csv(f"generated_test_data_{i + 1}.csv", index=False)


alloy_list = ['Al', 'Co', 'Cr', 'Cu', 'Fe', 'Ni']
# Shear Modulus (Pa)
g_Al = 2.4e10; g_Co = 8.26e10; g_Cr = 1.15e11; g_Cu = 4.6e10; g_Fe = 7.75e10; g_Ni = 7.6e10
# Cohesive Energy (kJ/mol)
ec_al = 3.27e5; ec_co = 4.24e5; ec_cr = 3.95e5; ec_cu = 3.36e5; ec_fe = 4.13e5; ec_ni = 4.28e5
# Work Function
w_al = 4.08; w_co = 5; w_cr = 5; w_cu = 4.7; w_fe = 4.5; w_ni = 5.01
# Valence Electron
vec_al = 3; vec_co = 9; vec_cr = 6; vec_cu = 11; vec_fe = 8; vec_ni = 10
# No of electrons
ea_al = 13; ea_co = 27; ea_cr = 24; ea_cu = 29; ea_fe = 26; ea_ni = 28

for i, data_set in enumerate(data_sets):
    df=pd.read_csv(f"generated_test_data_{i + 1}.csv")
    df['modulus_mismatch'] = 0

    df['shear_modulus'] = (df['Al'].astype('float') / 100 * g_Al + df['Co'].astype('float') / 100 * g_Co +
                           df['Cr'].astype('float') / 100 * g_Cr + df['Cu'].astype('float') / 100 * g_Cu +
                           df['Fe'].astype('float') / 100 * g_Fe + df['Ni'].astype('float') / 100 * g_Ni)
    for j in alloy_list:
        df['modulus_mismatch'] += (
                (df[f'{j}'].astype('float') / 100 * ((2 * (globals()[f'g_{j}'] -
                df['shear_modulus'].astype('float'))) / (globals()[f'g_{j}']) +
                df['shear_modulus'].astype('float'))) / (1 + (0.5 * abs(df[f'{j}'].astype('float')) * ((2 * (globals()[f'g_{j}'] -
                df['shear_modulus'].astype('float'))) / (globals()[f'g_{j}']) + df['shear_modulus']))))

    df['cohesive_energy'] = (df['Al'].astype('float') / 100 * ec_al + df['Co'].astype('float') / 100 * ec_co +
                             df['Cr'].astype('float') / 100 * ec_cr + df['Cu'].astype('float') / 100 * ec_cu +
                             df['Fe'].astype('float') / 100 * ec_fe + df['Ni'].astype('float') / 100 * ec_ni)

    df['6th_square_of_work_function'] = ((df['Al'].astype('float') / 100 * w_al) ** 6 +
                                         (df['Co'].astype('float') / 100 * w_co) ** 6 +
                                         (df['Cr'].astype('float') / 100 * w_cr) ** 6 +
                                         (df['Cu'].astype('float') / 100 * w_cu) ** 6 +
                                         (df['Fe'].astype('float') / 100 * w_fe) ** 6 +
                                         (df['Ni'].astype('float') / 100 * w_ni) ** 6)
    df['config_entropy'] = -1.5 * ((df['Al'].astype('float') / 100 * np.where(df['Al'].astype('float') != 0, np.log(df['Al'].astype('float') / 100), 0)) +
            (df['Co'].astype('float') / 100 * np.where(df['Co'].astype('float') != 0, np.log(df['Co'].astype('float') / 100), 0)) +
            (df['Cr'].astype('float') / 100 * np.where(df['Cr'].astype('float') != 0, np.log(df['Cr'].astype('float') / 100), 0)) +
            (df['Cu'].astype('float') / 100 * np.where(df['Cu'].astype('float') != 0, np.log(df['Cu'].astype('float') / 100), 0)) +
            (df['Fe'].astype('float') / 100 * np.where(df['Fe'].astype('float') != 0, np.log(df['Fe'].astype('float') / 100), 0)) +
            (df['Ni'].astype('float') / 100 * np.where(df['Ni'].astype('float') != 0, np.log(df['Ni'].astype('float') / 100), 0)))

    df['vec'] = (df['Al'].astype('float') / 100 * vec_al + df['Co'].astype('float') / 100 * vec_co + df['Cr'].astype('float') / 100 * vec_cr + df['Cu'].astype('float') / 100 * vec_cu + df['Fe'].astype('float') / 100 * vec_fe + df['Ni'].astype('float') / 100 * vec_ni)
    df['itinerant_electrons'] = (df['Al'].astype('float') / 100 * ea_al + df['Co'].astype('float') / 100 * ea_co + df['Cr'].astype('float') / 100 * ea_cr + df['Cu'].astype('float') / 100 * ea_cu + df['Fe'].astype('float') / 100 * ea_fe + df['Ni'].astype('float') / 100 * ea_ni)

    # LOGATHRMIC-FY LARGE TERM
    # df['shear_modulus'] = np.log10(df['shear_modulus']) # actually not used for training, but I'm doing it anyway
    df['shear_modulus'] = np.emath.logn(LOG_BASE, (df['shear_modulus']))  # actually not used for training, but I'm doing it anyway
    df['cohesive_energy'] = np.emath.logn(LOG_BASE, (df['cohesive_energy']))

    df['Hardness, HV'] = 0

    print(df)
    df.to_csv(f'./gen_test_extra_data_{i+1}.csv', index=False)


print(f"{num_splits} Excel files created successfully.")

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import logging
import os

# Set up logging
log_file = os.path.join(os.path.dirname(__file__), 'mobility.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')


center_address = './data/centers.dat' 
transfer_integral_address= './output/transfer_integrals.csv' 

# Constants and parameters
e = 1.602e-19  # Elementary charge in Coulombs C
k_B = 3.1668114e-6   # Boltzmann constant in Hartree/K
T = 300  # Temperature in Kelvin K
omega_0 = 1e4 #1e12  # Prefactor in Hz
F = (10e4)*(2.293712317e9)  # Electric field in Hartree/(Å·C) for 10e4 V/cm 1 V/cm = 2.293712317e9  Hartree/(Å·C) 
N = 80


# Remove columns from transfer_integrals.csv
# Function to remove columns after a specific column
def remove_columns_after(df, column_name):
    # Find the index of the specified column
    col_index = df.columns.get_loc(column_name)
    # Keep only columns up to and including the specified column
    return df.iloc[:, :col_index + 1]
# Read the CSV file
df_csv = pd.read_csv(transfer_integral_address)
# Remove columns after "Ej"
df_csv_filtered = remove_columns_after(df_csv, "Ej")
# Save the filtered dataframe back to a new CSV file
df_csv_filtered.to_csv(transfer_integral_address, index=False)



#Calc Rij and save
# Function to compute positive subsequent (absolute difference) between two numbers
def difference(a, b):
    return b-a

# Read the centers.dat file into a list of lists
with open(center_address, 'r') as centers_file: #./data/centers.dat'
    centers_data = [line.strip().split() for line in centers_file]

# Read the transfer_integrals CSV file
with open(transfer_integral_address, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    ti_lines = list(csvreader)

# Process each line in transfer_integrals starting from the second line (skipping the header)
for i, row in enumerate(ti_lines[1:], start=1):
    dimmer_name = row[0]
    if dimmer_name.startswith('dimmer'):
        # Extract the two numbers from the dimmer name
        parts = dimmer_name.split('_')
        index1 = int(parts[1])
        index2 = int(parts[2])

        # Get the lines corresponding to index1 and index2 from centers_data
        line1 = centers_data[index1]
        line2 = centers_data[index2]

        # Calculate positive subsequent for each column (1st, 2nd, 3rd)
        RijX = difference(float(line1[0]), float(line2[0]))
        RijY = difference(float(line1[1]), float(line2[1]))
        RijZ = difference(float(line1[2]), float(line2[2]))

        # Append the calculated values to the current line
        row.extend([f"{RijX}", f"{RijY}", f"{RijZ}"])

# Update the header of transfer_integrals file
ti_lines[0].extend(['RijX', 'RijY', 'RijZ'])

# Write the updated lines back to the transfer_integrals.csv file
with open(transfer_integral_address, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(ti_lines)

logging.info("transfer_integrals.csv file updated successfully.")




#Calc Wij and save

# Read the input file
data = pd.read_csv(transfer_integral_address)

# Ensure necessary columns are present
required_columns = ['HOMO', 'Ei', 'Ej', 'RijX', 'RijY', 'RijZ']
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The input file must contain the columns: {', '.join(required_columns)}")

# Calculate the hopping rates without storing intermediate results
hopping_rate_i_jx = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ej'] - data['Ei'] - e * F * data['RijX']) / (k_B * T))
hopping_rate_j_ix = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ei'] - data['Ej'] + e * F * data['RijX']) / (k_B * T))
hopping_rate_i_jy = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ej'] - data['Ei'] - e * F * data['RijY']) / (k_B * T))
hopping_rate_j_iy = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ei'] - data['Ej'] + e * F * data['RijY']) / (k_B * T))
hopping_rate_i_jz = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ej'] - data['Ei'] - e * F * data['RijZ']) / (k_B * T))
hopping_rate_j_iz = omega_0 * (data['HOMO'] ** 2) * np.exp(-(data['Ei'] - data['Ej'] + e * F * data['RijZ']) / (k_B * T))

# Add the hopping rates to the DataFrame
data['Wijx'] = hopping_rate_i_jx
data['Wjix'] = hopping_rate_j_ix
data['Wijy'] = hopping_rate_i_jy
data['Wjiy'] = hopping_rate_j_iy
data['Wijz'] = hopping_rate_i_jz
data['Wjiz'] = hopping_rate_j_iz
# Write the results back to the file
data.to_csv(transfer_integral_address, index=False)

logging.info("Hopping rates have been calculated and written to the file.")



#Load Rij and Wij from csv

# Step 1: Create an 80x80 matrix filled with zeros
omega_matrix_X = np.zeros((N, N), dtype=float)
omega_matrix_Y = np.zeros((N, N), dtype=float)
omega_matrix_Z = np.zeros((N, N), dtype=float)

RijX_matrix = np.zeros((N, N), dtype=float)
RijY_matrix = np.zeros((N, N), dtype=float)
RijZ_matrix = np.zeros((N, N), dtype=float)

# Step 2: Read the transfer_integrals.csv file and process it
# Open the file and read it line by line
with open(transfer_integral_address, 'r') as file:
    # Skip the header
    next(file)
    # Process each line
    for line in file:
        # Split the line by comma to get the relevant parts
        parts = line.split(',')
        filename = parts[0]
        omega_valueijX = float(parts[10])  # The sixth part is the value to be placed in the matrix
        omega_valuejiX = float(parts[11])
        omega_valueijY = float(parts[12])
        omega_valuejiY = float(parts[13])
        omega_valueijZ = float(parts[14])
        omega_valuejiZ = float(parts[15])
        RijX_value = float(parts[7])
        RijY_value = float(parts[8])
        RijZ_value = float(parts[9])
        
        # Split the filename by '_' and extract the second and third parts
        name_parts = filename.split('_')
        i, j = int(name_parts[1]), int(name_parts[2])
        # Update the matrix at position (i, j)
        omega_matrix_X[i, j] = omega_valueijX
        omega_matrix_X[j, i] = omega_valuejiX
        omega_matrix_Y[i, j] = omega_valueijY
        omega_matrix_Y[j, i] = omega_valuejiY
        omega_matrix_Z[i, j] = omega_valueijZ
        omega_matrix_Z[j, i] = omega_valuejiZ
        RijX_matrix[i, j] = RijX_value
        RijX_matrix[j, i] = -1 * RijX_value
        RijY_matrix[i, j] = RijY_value
        RijY_matrix[j, i] = -1 * RijY_value
        RijZ_matrix[i, j] = RijZ_value
        RijZ_matrix[j, i] = -1* RijZ_value



#cal p_i and_mobility
def calculate_p_i_and_mobility(omega, R, F, epsilon=1e-6, tol_p=1e-8, tol_mu=1e-9, max_iter=10000, seed=None):
    num_nodes = omega.shape[0]
    Nt = num_nodes
    if seed is not None:
        np.random.seed(seed)
    p = np.random.rand(num_nodes)/num_nodes  # initialize p_i randomly
    p_avg = np.mean(p)
    mu = 0
    p_avg_list = []
    mu_list = []
    
    for iteration in range(max_iter):
        p_old = p.copy()
        mu_old = mu
        
        # Update p_i
        for i in range(num_nodes):
            sum1 = np.sum([omega[i, j] * p[j] for j in range(num_nodes)])
            sum2 = np.sum([omega[k, i] for k in range(num_nodes)])
            sum3 = np.sum([(omega[j, i]-omega[i, j]) * p[j] for j in range(num_nodes)])
            p[i] = sum1 / ((sum2+epsilon) * (1 - (sum3 / (sum2+epsilon))))

        # Update mobility
        p = p / np.sum(p)
        p_avg = np.mean(p)
        mu = np.sum([omega[i, j] * p[i] * (1 - p[j]) * (R[i, j]) for i in range(num_nodes) for j in range(num_nodes)]) / (Nt * p_avg * F)
        mu = mu * 2.29371246e-3   
        # Store values for plotting
        p_avg_list.append(p_avg)
        mu_list.append(mu)
        if (iteration+1)%500 == 0:
            logging.info(f'Iteration {iteration+1}, Average p: {p_avg} | Mobility: {mu}')
            logging.info(f'                  tol_p: {np.linalg.norm(p - p_old)} | tol_mu: {abs(mu - mu_old)}')
            
        # Convergence check
        if np.linalg.norm(p - p_old) < tol_p and abs(mu - mu_old) < tol_mu:
            logging.info(f"Converged in iteration {iteration+1}!")
            logging.info(f'Final Iteration {iteration+1}, Average p: {p_avg} | Mobility: {mu}')
            break
    return p, mu, p_avg_list, mu_list

def plot_p_avg_mu(mobility, p_avg_list, mu_list):
    # Plot p_avg and mu over iterations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(p_avg_list, label='Average p')
    plt.xlabel('Iteration')
    plt.ylabel('p_avg')
    plt.title('Average Occupation Probability over Iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mu_list, label='Mobility')
    plt.xlabel('Iteration')
    plt.ylabel('Mobility m² ⁄ (V·s)')
    plt.title('Mobility over Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()  
    return



# Chosen hopping rates between nodes and distances between nodes
# Calculate occupation probabilities and mobility
p_X, mobility_X, p_avg_list_X, mu_list_X = calculate_p_i_and_mobility(omega_matrix_X, RijX_matrix, F, 
                                                              epsilon=1e-16, tol_p=1e-7, tol_mu=1e-9, 
                                                              max_iter=int(1e7), seed=0)


p_Y, mobility_Y, p_avg_list_Y, mu_list_Y = calculate_p_i_and_mobility(omega_matrix_Y, RijY_matrix, F, 
                                                              epsilon=1e-16, tol_p=1e-7, tol_mu=1e-9, 
                                                              max_iter=int(1e7), seed=0)


p_Z, mobility_Z, p_avg_list_Z, mu_list_Z = calculate_p_i_and_mobility(omega_matrix_Z, RijZ_matrix, F, 
                                                              epsilon=1e-16, tol_p=1e-7, tol_mu=1e-9, 
                                                              max_iter=int(1e7), seed=0)

# Print P
logging.info(f"P in X direction: {p_X}")
logging.info(f"P in Y direction: {p_Y}")
logging.info(f"P in Z direction: {p_Z}")     
      
# Print computed mobilities
logging.info(f"Mobility in X direction: {mobility_X} m²/(V·s)")
logging.info(f"Mobility in Y direction: {mobility_Y} m²/(V·s)")
logging.info(f"Mobility in Z direction: {mobility_Z} m²/(V·s)")
                                                                                                                  

import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D 

############################################################
#parameters
PRINT = False 
PLOT = False 
threshold=15
centers_directory = './data'
centers_filename = 'centers.dat'
coords_directory = './data'
coords_name = 'coords'
dimmers_name = 'dimmers'
sbatch_folder_name = 'sbatch'
output_folder_name = 'output'
############################################################

def load_centers(directory='./data', filename='centers.dat'):
    # Construct the full path
    full_path = directory + '/' + filename
    try:
        # Load data from file
        data = np.loadtxt(full_path)
        return data
    except Exception as e:
        print("An error occurred in load_centers step:", e)
        return None
    
def compute_neighbors(centers):
    # Compute Delaunay triangulation
    triangulation = Delaunay(centers)
    # Get indices and neighbor information
    indptr_neigh, neighbors = triangulation.vertex_neighbor_vertices
    # Create a dictionary to store neighbors
    neighbor_dict = {}
    # Accessing the neighbors
    for i in range(len(centers)):
        i_neighbors = neighbors[indptr_neigh[i]:indptr_neigh[i+1]]
        neighbor_dict[i] = i_neighbors.tolist()
    return neighbor_dict

def Plot_Voronoi_3D(points):
    # Delaunay triangulation
    tri = Delaunay(points)  # Delaunay
    # Voronoi diagram
    vor = Voronoi(points)  # Voronoi
    # Plot
    fig = plt.figure(figsize=(10, 5))
    # Delaunay triangulation
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles=tri.simplices, color='skyblue', edgecolor='k')
    ax1.scatter(points[:,0], points[:,1], points[:,2], c='red', s=50)
    ax1.set_title('Delaunay Triangulation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # Voronoi diagram
    ax2 = fig.add_subplot(122, projection='3d')
    for simplex in tri.simplices:
        simplex = np.append(simplex, simplex[0])  # Closed loop
        ax2.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'k-')
    ax2.scatter(points[:,0], points[:,1], points[:,2], c='red', s=50)
    ax2.set_title('Voronoi Diagram')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
    return

def Plot_Voronoi_2D(points):
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Delaunay triangulation
    tri = Delaunay(points)
    ax1.triplot(points[:,0], points[:,1], tri.simplices)
    ax1.plot(points[:,0], points[:,1], 'o')
    for i, p in enumerate(points):
        ax1.text(p[0], p[1], str(i), ha='center')
    ax1.set_title('Delaunay Triangulation')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Voronoi diagram
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax2)
    for i, p in enumerate(points):
        ax2.text(p[0], p[1], str(i), ha='center')
    ax2.set_title('Voronoi Diagram')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.show()
    return

def compute_distance_matrix(neighbor_dict, centers):
    num_nodes = len(neighbor_dict)
    distance_matrix = np.zeros((num_nodes, num_nodes))
    for key, neighbors in neighbor_dict.items():
        for neighbor in neighbors:
            distance = np.linalg.norm(centers[key] - centers[neighbor])
            distance_matrix[key, neighbor] = distance
    return distance_matrix

def filter_distance_matrix(distance_matrix, threshold=15):
    filtered_matrix = np.where(distance_matrix > threshold, 0, distance_matrix)
    return filtered_matrix

def get_nonzero_indices(matrix):
    # Get the upper triangular part of the matrix
    upper_triangular = np.triu(matrix, k=1)
    # Get the indices of non-zero elements
    nonzero_indices = np.transpose(np.nonzero(upper_triangular))
    # Adjust indices to consider the lower triangular part
    nonzero_indices = [(i, j) if i < j else (j, i) for i, j in nonzero_indices]
    return nonzero_indices

def create_dimmer_sbatch_files(pair_indices, dimmers_name, coords_name, coords_directory, output_folder_name, sbatch_folder_name):
    dimmers_directory = './'+ dimmers_name
    output_directory = './'+ output_folder_name
    sbatch_directory = './'+ sbatch_folder_name
    if not os.path.exists(dimmers_directory):
        os.makedirs(dimmers_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if not os.path.exists(sbatch_directory):
        os.makedirs(sbatch_directory)
    i=0
    for pair in pair_indices:
        i+=1
        x_index, y_index = pair
        
        # Load coords_x.dat
        x_coords_file = os.path.join(coords_directory, f'{coords_name}_{x_index}.dat')
        with open(x_coords_file, 'r') as f:
            x_coords = f.readlines()
            
        # Load coords_y.dat
        y_coords_file = os.path.join(coords_directory, f'{coords_name}_{y_index}.dat')
        with open(y_coords_file, 'r') as f:
            y_coords = f.readlines()
        
        start = ["%nprocshared=32\n",
                 "%mem=90GB\n", 
                 f"%chk=../{dimmers_name}/{dimmers_name}_{x_index}_{y_index}.chk\n", 
                 "# wb97xd/6-31g(d)\n", 
                 "# symmetry=none\n", 
                 "\n", 
                 "tcal\n", 
                 "\n",
                 "0 1\n"]
        end = ["\n",
                "\n", 
                "--Link1--\n", 
                "%nprocshared=32\n",
                "%mem=90GB\n",
                f"%chk=../{dimmers_name}/{dimmers_name}_{x_index}_{y_index}.chk\n", 
                "# wb97xd/6-31g(d)\n", 
                "# symmetry=none\n", 
                "# geom=allcheck\n", 
                "# guess=read\n", 
                "# pop=full\n", 
                "# iop(3/33=4,5/33=3)\n"]
        
        # Concatenate the coordinates
        dimmer_coords = start+ x_coords + y_coords + end
        # Write concatenated coordinates to a new file
        dimmer_file = os.path.join(dimmers_directory, f'{dimmers_name}_{x_index}_{y_index}.gjf')
        with open(dimmer_file, 'w') as f:
            f.writelines(dimmer_coords)
        sbatch =   ["#!/bin/bash\n", 
                    "\n",    
                    "#SBATCH --time=2:00:00\n",                
                    "#SBATCH -N1\n",
                    "#SBATCH -A Your Account Number\n",
                    f'#SBATCH --job-name="{dimmers_name}_{x_index}_{y_index}"\n',
                    f"#SBATCH  -o ../{output_folder_name}/{dimmers_name}_{x_index}_{y_index}.out\n", 
                    "\n",
                    "module load Your Modules\n",
                    f"python ../tcal.py -a ../{dimmers_name}/{dimmers_name}_{x_index}_{y_index}.gjf\n"]       
        # Write sbatch to a new file
        sbatch_file = os.path.join(sbatch_directory, f'{i}.sh')
        with open(sbatch_file, 'w') as f:
            f.writelines(sbatch)
    array =   ["#!/bin/bash\n", 
                "\n",    
                "#SBATCH --time=02:00:00\n",                
                "#SBATCH -N1\n",
                "#SBATCH -A Your Account Number\n",
                f'#SBATCH --array 1-{i}\n',
                "#actual_task_id=$((SLURM_ARRAY_TASK_ID + 0))\n", 
                "bash $actual_task_id\n"]
    array_file = os.path.join(sbatch_directory, 'array.sh')
    with open(array_file, 'w') as f:
        f.writelines(array)
    return
          
def preprocess(PRINT,PLOT,threshold,centers_directory,centers_filename,coords_directory,
               coords_name,dimmers_name,sbatch_folder_name,output_folder_name):
    # load centers
    centers_data = load_centers(centers_directory, centers_filename)
    if centers_data is not None:
        if PRINT:
            print(f"Centers read from {centers_directory}/{centers_filename}:")
            print(centers_data)
    else:
        raise Exception(f"Failed to load centers from {centers_directory}/{centers_filename}.")

    # compute neighbors
    neighbors = compute_neighbors(centers_data)
    if PRINT:
        print(neighbors)
    if PLOT:
        Plot_Voronoi_3D(centers_data)

    # compute distance of neighbors
    distance_matrix = compute_distance_matrix(neighbor_dict=neighbors, centers=centers_data)
    if PRINT:
        print("Distance Matrix:")
        print(distance_matrix)

    # filter neighbors
    filtered_distance_matrix = filter_distance_matrix(distance_matrix, threshold)
    if PRINT:
        print("Filtered Distance Matrix (Threshold =", threshold, "):")
        print(filtered_distance_matrix)

    # Get unique neighbor pairs
    neighbor_pairs = get_nonzero_indices(filtered_distance_matrix)
    if PRINT:
        print(f"Neighbor Pairs in total {len(neighbor_pairs)}:")
        print(neighbor_pairs)

    #Making Dimer of Pairs & Sbatch for each pair starting from 1 to length & Array
    create_dimmer_sbatch_files(pair_indices = neighbor_pairs, dimmers_name=dimmers_name, 
                        coords_name=coords_name, coords_directory=coords_directory, 
                        output_folder_name=output_folder_name, sbatch_folder_name = sbatch_folder_name)
    return

preprocess(PRINT,PLOT,threshold,centers_directory,centers_filename,coords_directory,
               coords_name,dimmers_name,sbatch_folder_name,output_folder_name)

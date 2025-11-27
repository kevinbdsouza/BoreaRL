import numpy as np

def compute_hypervolume_2d(points: np.ndarray, ref_point: np.ndarray = np.array([-2, -2])) -> float:
    """
    Compute 2D hypervolume with reference point (-2, -2).
    """
    if len(points) == 0:
        return 0.0
    
    # Filter points that dominate reference point
    valid_points = points[(points[:, 0] > ref_point[0]) & (points[:, 1] > ref_point[1])]
    if len(valid_points) == 0:
        return 0.0
    
    # Sort by first objective (carbon)
    sorted_points = valid_points[np.argsort(valid_points[:, 0])]
    
    # Compute hypervolume
    hv = 0.0
    prev_x = ref_point[0]
    
    for point in sorted_points:
        if point[0] > prev_x:
            width = point[0] - prev_x
            height = point[1] - ref_point[1]
            hv += width * height
            prev_x = point[0]
    
    return hv

def compute_sparsity(points: np.ndarray) -> float:
    """
    Compute Sparsity (Spacing) metric.
    S = sqrt( (1/(n-1)) * sum( (d_i - d_mean)^2 ) )
    where d_i is the distance between consecutive points in the sorted front.
    Lower is better (more uniform distribution).
    """
    if len(points) < 2:
        return 0.0
        
    # Sort by first objective
    sorted_points = points[np.argsort(points[:, 0])]
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(sorted_points) - 1):
        p1 = sorted_points[i]
        p2 = sorted_points[i+1]
        # Euclidean distance
        dist = np.sqrt(np.sum((p1 - p2)**2))
        distances.append(dist)
        
    if not distances:
        return 0.0
        
    # Calculate standard deviation of distances
    # Note: The standard Spacing metric formula is essentially the standard deviation 
    # of these distances.
    s = np.std(distances, ddof=1) # Use sample standard deviation
    
    return s

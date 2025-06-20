import numpy as np
import math

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """
    Find the intersection points between a circle and a line segment.
    
    Args:
        circle_center: Center of the circle (x, y)
        circle_radius: Radius of the circle
        pt1: First point of the line segment (x, y)
        pt2: Second point of the line segment (x, y)
        full_line: If True, treat as infinite line; if False, treat as line segment
        tangent_tol: Tolerance for tangent lines
    
    Returns:
        List of intersection points
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    
    # Translate to put circle at origin
    p1x -= cx
    p1y -= cy
    p2x -= cx
    p2y -= cy
    
    # Calculate line coefficients
    dx = p2x - p1x
    dy = p2y - p1y
    dr = math.sqrt(dx**2 + dy**2)
    D = p1x * p2y - p2x * p1y
    
    # Calculate discriminant
    discriminant = circle_radius**2 * dr**2 - D**2
    
    if discriminant < 0:
        return []  # No intersection
    
    # Calculate intersection points
    intersections = []
    
    if discriminant == 0:
        # One intersection (tangent)
        x = D * dy / (dr**2)
        y = -D * dx / (dr**2)
        intersections.append((x + cx, y + cy))
    else:
        # Two intersections
        sqrt_discriminant = math.sqrt(discriminant)
        sign_dy = 1 if dy >= 0 else -1
        
        x1 = (D * dy + sign_dy * dx * sqrt_discriminant) / (dr**2)
        y1 = (-D * dx + abs(dy) * sqrt_discriminant) / (dr**2)
        
        x2 = (D * dy - sign_dy * dx * sqrt_discriminant) / (dr**2)
        y2 = (-D * dx - abs(dy) * sqrt_discriminant) / (dr**2)
        
        intersections.extend([(x1 + cx, y1 + cy), (x2 + cx, y2 + cy)])
    
    if not full_line:
        # Filter to only include points on the line segment
        fractions = []
        for x, y in intersections:
            # Convert back to original coordinate system
            x -= cx
            y -= cy
            
            if abs(dx) > abs(dy):
                fraction = (x - p1x) / dx if dx != 0 else 0
            else:
                fraction = (y - p1y) / dy if dy != 0 else 0
            
            fractions.append(fraction)
        
        # Keep only intersections that lie on the segment (0 <= fraction <= 1)
        intersections = [pt for pt, frac in zip(intersections, fractions) 
                        if 0 <= frac <= 1]
    
    return intersections


def get_target_point(lookahead, polyline):
    """
    Get the target point for pure pursuit control.
    
    Args:
        lookahead: Lookahead distance
        polyline: Array of waypoints [(x1, y1), (x2, y2), ...]
    
    Returns:
        Target point (x, y) or None if no intersection found
    """
    if len(polyline) < 2:
        return None
        
    intersections = []
    
    # Check intersection with each line segment of the polyline
    for j in range(len(polyline) - 1):
        pt1 = polyline[j]
        pt2 = polyline[j + 1]
        
        # Find intersections with circle centered at origin (vehicle position)
        segment_intersections = circle_line_segment_intersection(
            (0, 0), lookahead, pt1, pt2, full_line=False
        )
        intersections.extend(segment_intersections)
    
    # Filter intersections to only include those ahead of the vehicle (positive x)
    filtered = [p for p in intersections if p[0] > 0]
    
    if len(filtered) == 0:
        return None
    
    # Return the first intersection point (closest to vehicle in terms of path progress)
    return filtered[0]


def get_curvature(trajectory):
    """
    Calculate the maximum curvature of a trajectory.
    
    Args:
        trajectory: Array of points [(x1, y1), (x2, y2), ...]
    
    Returns:
        Maximum curvature value
    """
    if len(trajectory) < 3:
        return 0.0
    
    curvatures = []
    
    for i in range(1, len(trajectory) - 1):
        # Get three consecutive points
        p1 = np.array(trajectory[i-1])
        p2 = np.array(trajectory[i])
        p3 = np.array(trajectory[i+1])
        
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Calculate cross product and dot product
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)
        
        # Calculate curvature using the formula: k = |v1 × v2| / |v1|^3
        v1_mag = np.linalg.norm(v1)
        v2_mag = np.linalg.norm(v2)
        
        if v1_mag > 1e-6 and v2_mag > 1e-6:
            # Use the formula: k = 2 * sin(θ) / |chord|
            # where θ is the angle between v1 and v2
            cos_theta = dot / (v1_mag * v2_mag)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Clamp to avoid numerical errors
            theta = np.arccos(cos_theta)
            
            chord_length = np.linalg.norm(p3 - p1)
            if chord_length > 1e-6:
                curvature = 2 * np.sin(theta) / chord_length
                curvatures.append(abs(curvature))
    
    return max(curvatures) if curvatures else 0.0 
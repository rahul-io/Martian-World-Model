import numpy as np
import scipy
from utils.camera_utils import viewmatrix, normalize, pad_poses, unpad_poses, poses_to_points, points_to_poses, interp

def generate_interpolated_path(poses, n_interp, method="spline", 
                               spline_degree=5, smoothness=.03, rot_weight=.1, 
                               ellipse_factor=(0.5, 0.2), depth=None):
    p0 = poses[0][:3, -1]
    p1 = poses[1][:3, -1]
    
    view0 = -poses[0][:3, 2]
    view1 = -poses[1][:3, 2]
    viewdir = normalize((view0 + view1) / 2.0)

    orig_up = normalize((poses[0][:3, 1] + poses[1][:3, 1]) / 2.0)
    up_mod = orig_up - np.dot(orig_up, viewdir) * viewdir
    up_mod = normalize(up_mod)

    if method == "circle":
        return generate_circle_trajectory(poses, n_interp, viewdir, up_mod, rot_weight)
    elif method == "zoom_in_out":
        return generate_zoom_trajectory(poses, n_interp, rot_weight, viewdir, up_mod)
    elif method == "falling_leaf":
        return generate_falling_leaf_trajectory(poses, n_interp, rot_weight, viewdir, up_mod)
    elif method == "ellipse":
        return generate_ellipse_trajectory(poses, n_interp, rot_weight, viewdir, up_mod, ellipse_factor)
    elif method == "spiral":
        return generate_spiral_trajectory(poses, n_interp, rot_weight, viewdir, up_mod)
    elif method == "parabolic":
        return generate_parabolic_trajectory(poses, n_interp, rot_weight, viewdir, up_mod, depth)
    else:
        return generate_spline_trajectory(poses, n_interp, rot_weight, spline_degree, smoothness)

def generate_circle_trajectory(poses, n_interp, viewdir, up_mod, rot_weight, num_loops=7):
    if poses.shape[0] < 2:
        raise ValueError("circle trajectory requires at least two keyframe poses")
    
    p0 = poses[0][:3, -1]
    p1 = poses[1][:3, -1]
    view0 = -poses[0][:3, 2]
    view1 = -poses[1][:3, 2]

    forward_offset = rot_weight * 0.1
    radius_ratio = 0.3

    p0_forward = p0 + forward_offset * view0
    p1_forward = p1 + forward_offset * view1

    center = (p0_forward + p1_forward) / 2.0

    def project_to_plane(point, plane_point, plane_normal):
        return plane_point + (point - plane_point) - np.dot(point - plane_point, plane_normal) * plane_normal

    p0_proj = project_to_plane(p0_forward, center, viewdir)
    p1_proj = project_to_plane(p1_forward, center, viewdir)

    r0 = np.linalg.norm(p0_proj - center)
    r1 = np.linalg.norm(p1_proj - center)
    if r0 < 1e-6 or r1 < 1e-6:
        radius = np.linalg.norm(p1 - p0) / 2.0
    else:
        radius = (r0 + r1) / 2.0

    if np.linalg.norm(p0_proj - center) < 1e-6:
        u = normalize(np.cross(viewdir, np.array([1, 0, 0])))
        if np.linalg.norm(u) < 1e-6:
            u = normalize(np.cross(viewdir, np.array([0, 1, 0])))
    else:
        u = normalize(p0_proj - center)
    v = normalize(np.cross(viewdir, u))

    vec_p1 = p1_proj - center
    theta1 = np.arctan2(np.dot(vec_p1, v), np.dot(vec_p1, u))

    angles = np.linspace(0, theta1 + 2 * np.pi * num_loops, n_interp)

    poses_list = []
    for theta in angles:
        pos = center + radius_ratio * radius * (np.cos(theta) * u + np.sin(theta) * v)
        lookat = pos - viewdir
        cam_matrix = viewmatrix(lookat - pos, up_mod, pos)
        poses_list.append(cam_matrix)

    return np.stack(poses_list, axis=0)

def generate_zoom_trajectory(poses, n_interp, rot_weight, viewdir, up_mod):
    p0 = poses[0][:3, -1]
    p1 = poses[1][:3, -1]
    view0 = -poses[0][:3, 2] 
    view1 = -poses[1][:3, 2] 
    center = (p0 + p1) / 2.0
    target = center + 0.4 * rot_weight * viewdir

    poses_list = []
    n1 = n_interp // 2
    n2 = n_interp - n1

    for t in np.linspace(0, 1, n1, endpoint=False):
        pos = (1 - t) * p0 + t * target
        interp_view = normalize((1 - t) * view0 + t * viewdir)
        lookat = pos - interp_view
        cam_matrix = viewmatrix(lookat - pos, up_mod, pos)
        poses_list.append(cam_matrix)

    for t in np.linspace(0, 1, n2, endpoint=True):
        pos = (1 - t) * target + t * p1
        interp_view = normalize((1 - t) * viewdir + t * view1)
        lookat = pos - interp_view
        cam_matrix = viewmatrix(lookat - pos, up_mod, pos)
        poses_list.append(cam_matrix)
            
    return np.stack(poses_list, axis=0)

def generate_falling_leaf_trajectory(poses, n_interp, rot_weight, viewdir, up_mod):
    if poses.shape[0] < 2:
        raise ValueError("need at least two keyframes for falling_leaf trajectory")
        
    p0 = poses[0][:3, -1]
    p1 = poses[1][:3, -1]
    view0 = -poses[0][:3, 2] 
    view1 = -poses[1][:3, 2] 

    t_vals = np.linspace(0, 1, n_interp)
    
    pos_base = (1 - t_vals)[:, None] * p0 + t_vals[:, None] * p1
    
    base_views = np.array([normalize((1 - t) * view0 + t * view1) for t in t_vals])
    
    d = p1 - p0
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-6:
        raise ValueError("keyframe positions are too close")
    d_dir = d / d_norm
    lateral_dir = normalize(np.cross(up_mod, d_dir))

    amplitude_base = 0.2 * d_norm * rot_weight

    swing = (1 - t_vals) * np.sin(2 * np.pi * 2.5 * t_vals)
    final_positions = pos_base + amplitude_base * swing[:, None] * lateral_dir

    final_views = []
    for i, t in enumerate(t_vals):
        s = np.sin(2 * np.pi * 2.5 * t)
        frac = abs(s)
        if s >= 0:
            corrected_view = normalize((1 - frac) * base_views[i] + frac * view1)
        else:
            corrected_view = normalize((1 - frac) * base_views[i] + frac * view0)
        final_views.append(corrected_view)
    final_views = np.array(final_views)
    
    poses_list = []
    for i in range(n_interp):
        pos = final_positions[i]
        cam_matrix = viewmatrix(-final_views[i], up_mod, pos)
        poses_list.append(cam_matrix)
    
    return np.stack(poses_list, axis=0)

def generate_ellipse_trajectory(poses, n_interp, rot_weight, viewdir, up_mod, ellipse_factor):
    if poses.shape[0] < 2:
        raise ValueError("ellipse trajectory requires at least two keyframe poses")
    
    p0_orig = poses[0][:3, -1]
    p1_orig = poses[1][:3, -1]
    
    view0 = -poses[0][:3, 2]
    view1 = -poses[1][:3, 2]
    
    p0_forward = p0_orig + 0.2 * rot_weight * view0
    p1_forward = p1_orig + 0.2 * rot_weight * view1
    
    center = (p0_forward + p1_forward) / 2.0
    
    axis_major = p1_forward - p0_forward
    axis_major_length = np.linalg.norm(axis_major)
    if axis_major_length < 1e-6:
        raise ValueError("keyframe positions are too close to construct ellipse trajectory.")

    a = (axis_major_length / 2.0) * ellipse_factor[0]
    axis_major_dir = normalize(axis_major)  
    axis_minor_dir = np.cross(viewdir, axis_major_dir)
    axis_minor_dir = normalize(axis_minor_dir)
    b = a * ellipse_factor[1]  
    
    angles = np.linspace(0, 4 * 2 * np.pi, n_interp, endpoint=False)
    poses_list = []
    for ang in angles:
        pos = center + a * np.cos(ang) * axis_major_dir + b * np.sin(ang) * axis_minor_dir
        t_interp = (np.sin(ang) + 1) / 2.0
        final_view = normalize((1 - t_interp) * view0 + t_interp * view1)
        cam_matrix = viewmatrix(-final_view, up_mod, pos)
        poses_list.append(cam_matrix)
    
    return np.stack(poses_list, axis=0)

def generate_spiral_trajectory(poses, n_interp, rot_weight, viewdir, up_mod):
    if len(poses) < 2:
        raise ValueError("spiral trajectory requires at least two keyframe poses")

    p0 = poses[0][:3, -1]
    p1 = poses[1][:3, -1]
    mid_point = (p0 + p1) / 2.0

    vec0_proj = (p0 - mid_point) - np.dot(p0 - mid_point, viewdir)*viewdir
    vec1_proj = (p1 - mid_point) - np.dot(p1 - mid_point, viewdir)*viewdir
    base_radius = max((np.linalg.norm(vec0_proj) + np.linalg.norm(vec1_proj))/2.0, 0.5 * rot_weight)

    if np.linalg.norm(vec0_proj) > 1e-3:
        x_axis = normalize(vec0_proj)
    else: 
        x_axis = normalize(np.cross(up_mod, viewdir))
    y_axis = normalize(np.cross(viewdir, x_axis))

    spiral_params = {
        'start_offset' : rot_weight * 0.1,  
        'total_height' : rot_weight * 0.6,  
        'num_rotations': 5,                
        'radius_curve' : lambda t: base_radius * (0.12 - 0.09 * t)
    }

    t_vals = np.linspace(0, 0.8, n_interp)
    heights = spiral_params['start_offset'] + t_vals * spiral_params['total_height']
    angles = 2 * np.pi * spiral_params['num_rotations'] * t_vals

    focus_point = mid_point - viewdir * (spiral_params['total_height'] + 100.0)

    poses_list = []
    for i in range(n_interp):
        current_radius = spiral_params['radius_curve'](t_vals[i])
        current_height = heights[i]
        
        spiral_x = current_radius * np.cos(angles[i])
        spiral_y = current_radius * np.sin(angles[i])
        
        position = mid_point + viewdir*current_height + spiral_x*x_axis + spiral_y*y_axis
        
        cam_matrix = viewmatrix(focus_point - position, up_mod, position)
        poses_list.append(cam_matrix)
    
    return np.stack(poses_list, axis=0)

def generate_parabolic_trajectory(
    poses: np.ndarray, 
    n_interp: int, 
    rot_weight: float, 
    viewdir: np.ndarray, 
    up_mod: np.ndarray, 
    depth: np.ndarray,
    bump_frequency_factor: float = 1.5, # Controls how fast the bumps change
    bump_amplitude_scale: float = 0.005, # Controls the magnitude of the bumps relative to scene scale
    view_oscillation_freq: float = 0.7 # How many times the view oscillates between start/end
    ) -> np.ndarray:
    if poses.shape[0] < 2:
        raise ValueError("At least two keyframe poses are required.")
    if n_interp <= 0:
        raise ValueError("n_interp must be positive.")

    # Ensure vectors are normalized
    viewdir = normalize(viewdir)
    up_mod = normalize(up_mod)
    
    # Extract start position and the two initial view directions
    p0 = poses[0][:3, 3] # Position from the 4x4 matrix
    p1 = poses[1][:3, 3] 
    # View direction is often the negative Z-axis of the camera pose matrix
    # C2W matrix: Z axis points out of camera back. We want direction camera *looks*
    # If poses are Camera-to-World (C2W):
    view0 = normalize(poses[0][:3, 2]) # Z-axis of camera in world coords
    view1 = normalize(poses[1][:3, 2]) # Z-axis of camera in world coords

    # -- Calculate Motion Scale based on Depth (if available) --
    d_keyframes = np.linalg.norm(p1 - p0)
    if d_keyframes < 1e-6:
       # Fallback if keyframes are too close, use a default scale or rot_weight
       print("Warning: Keyframe poses are very close. Using default scaling.")
       forward_distance = rot_weight * 1.0 # Default distance
       scene_scale_estimate = forward_distance 
    else:
       # Base forward distance on keyframe distance and rot_weight
       forward_distance = 0.4 * rot_weight * d_keyframes 

    avg_depth = None
    if depth is not None and len(depth) >= 2:
        try:
            depth0, depth1 = depth[0], depth[1]
            # Use median of valid depth values as robust estimate
            median_depth0 = np.median(depth0[depth0 > 0]) if np.any(depth0 > 0) else 1.0
            median_depth1 = np.median(depth1[depth1 > 0]) if np.any(depth1 > 0) else 1.0
            avg_depth = (median_depth0 + median_depth1) / 2.0
            
            # Calculate scale factor: ratio of avg depth to keyframe distance
            # This assumes the keyframe distance is somewhat representative of the scale
            scale_factor = avg_depth / d_keyframes
            # Clip scale factor to prevent extreme values
            scale_factor = np.clip(scale_factor, 0.5, 3.0) # Adjusted clipping range

            # Adjust forward distance based on depth scale
            forward_distance *= scale_factor
            scene_scale_estimate = avg_depth # Use depth for bump scale
            print(f"Depth info used: Avg Depth={avg_depth:.2f}, Scale Factor={scale_factor:.2f}")

        except Exception as e:
            print(f"Warning: Could not process depth info: {e}. Using distance-based scaling.")
            scene_scale_estimate = forward_distance # Fallback to distance
    else:
        scene_scale_estimate = forward_distance # No depth info, use distance

    # -- Generate Base Trajectory (Forward Motion) --
    t_vals = np.linspace(0, 1, n_interp)
    base_positions = p0[None, :] + np.outer(t_vals, forward_distance * viewdir)

    # -- Generate Smooth Random Bumps ---
    # Calculate lateral (right) direction
    lateral_dir = normalize(np.cross(viewdir, up_mod)) 

    # Number of control points for smooth interpolation of bumps
    # More points = smoother but potentially more computation
    num_control_points = max(3, int(n_interp / (10 / bump_frequency_factor))) # Make freq adjustable
    control_times = np.linspace(0, 1, num_control_points)

    # Amplitude of bumps, scaled by scene estimate and user factor
    bump_amplitude = scene_scale_estimate * bump_amplitude_scale 
    
    # Generate random offsets at control points
    # Smaller vertical bumps often feel more natural
    horizontal_control = bump_amplitude * np.random.uniform(-1, 1, size=num_control_points)
    vertical_control = (0.5 * bump_amplitude) * np.random.uniform(-1, 1, size=num_control_points)
    
    # Interpolate offsets for all frames
    horizontal_offsets = np.interp(t_vals, control_times, horizontal_control)
    vertical_offsets = np.interp(t_vals, control_times, vertical_control)

    # Add bumps to the base trajectory
    final_positions = base_positions + horizontal_offsets[:, None] * lateral_dir + vertical_offsets[:, None] * up_mod

    # -- Generate Oscillating View Direction using SLERP --
    # Create Scipy Rotation objects. We need to represent the directions.
    # A common way is aligning a reference vector (e.g., [0,0,1]) to the target direction.
    # Ensure view vectors are valid directions
    if np.linalg.norm(view0) < 1e-6 or np.linalg.norm(view1) < 1e-6:
        print("Warning: Invalid view direction(s), using average viewdir.")
        final_views = np.tile(viewdir, (n_interp, 1))
    else:
        try:
            # Use Rotation.align_vectors to find rotation mapping [0,0,1] to view vectors
            # Note: This assumes view0/view1 are directions camera *looks along*
            # If view0/view1 represent the camera's Z-axis in world (pointing backward), 
            # you might need to negate them depending on your convention. 
            # Let's assume they are the LOOKING direction here.
            from scipy.spatial.transform import Rotation as R
            from scipy.spatial.transform import Slerp
            ref_vec = np.array([[0.0, 0.0, 1.0]]) # Reference vector
            rot0, _ = R.align_vectors(ref_vec, [view0])
            rot1, _ = R.align_vectors(ref_vec, [view1])
            key_rots = R.concatenate([rot0, rot1])
            key_times = [0, 1] # Corresponding to rot0 and rot1
            
            # Create the SLERP interpolator
            slerp_interpolator = Slerp(key_times, key_rots)

            # Generate oscillating interpolation weights (0 to 1 and back)
            # Cosine ensures it starts at view0 (w=0) and ends near view0/1 depending on freq.
            # w = 0.5 * (1 - np.cos(t_vals * 2 * np.pi * view_oscillation_freq)) # Starts at 0, ends at 0 if freq is integer
            # Use cosine starting from 1, so w oscillates between 0 and 1.
            w_oscillating = 0.5 * (1 + np.cos(t_vals * np.pi * view_oscillation_freq))
            w_oscillating = np.clip(w_oscillating, 0.1, 0.9)
            final_views = np.array([normalize((1 - w) * view0 + w * view1) for w in w_oscillating])
            
            # Interpolate rotations using SLERP with the oscillating weights
            interp_rots = slerp_interpolator(w_oscillating)
            
            # Apply the interpolated rotations to the reference vector to get view directions
            final_views = interp_rots.apply(ref_vec.flatten()) # Shape (n_interp, 3)
            final_views = normalize(final_views) # Ensure normalization

        except Exception as e:
            print(f"Warning: SLERP failed ({e}), falling back to average viewdir.")
            final_views = np.tile(viewdir, (n_interp, 1))


    poses_list = []
    for i in range(n_interp):
        pos = final_positions[i]
        view = final_views[i]
        # Use the viewmatrix helper function
        # Assumes viewmatrix takes (view_direction, up_vector, camera_position)
        cam_matrix = viewmatrix(view, up_mod, pos) 
        poses_list.append(cam_matrix)
        
    return np.stack(poses_list, axis=0)

def generate_spline_trajectory(poses, n_interp, rot_weight, spline_degree, smoothness):
    """Generate spline-based camera trajectory"""
    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)

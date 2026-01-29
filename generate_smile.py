
import numpy as np
from obj_loader import load_obj, calculate_deltas

def generate_smile_mesh(neutral_path, open_mouth_path, output_path):
    print(f"Loading {neutral_path} and {open_mouth_path}...")
    neutral_mesh = load_obj(neutral_path)
    open_mouth_mesh = load_obj(open_mouth_path)
    
    if not neutral_mesh or not open_mouth_mesh:
        return
        
    # Calculate delta to find mouth region
    # Open mouth usually moves the jaw down and lips apart.
    delta_pos, _ = calculate_deltas(neutral_mesh, open_mouth_mesh)
    
    # Identify mouth vertices: those that moved significantly > 0.005
    moved_indices = []
    for i, d in enumerate(delta_pos):
        if np.linalg.norm(d) > 0.005:
            moved_indices.append(i)
            
    print(f"Found {len(moved_indices)} vertices in the mobile region (likely mouth/jaw).")
    
    # Calculate center of this region to distinguish upper/lower lip if possible, or just apply field
    moved_coords = neutral_mesh.vertices[moved_indices]
    min_bounds = np.min(moved_coords, axis=0)
    max_bounds = np.max(moved_coords, axis=0)
    center_bounds = (min_bounds + max_bounds) / 2.0
    
    print(f"Mouth Region Bounds: Min={min_bounds}, Max={max_bounds}")
    print(f"Region Center: {center_bounds}")
    
    # Create new vertices
    new_vertices = neutral_mesh.vertices.copy()
    
    # Parametric Smile
    # Formula: Widen (x away from 0) and Lift (y up) based on distance from center X
    # We want to effect mainly the corners.
    
    # Rough approximation of mouth corners:
    # Max X and Min X in the moved region.
    
    for i in moved_indices:
        v = neutral_mesh.vertices[i]
        x = v[0]
        y = v[1]
        z = v[2]
        
        # Distance from center X (horizontal)
        dist_x = abs(x - center_bounds[0]) # Assuming center X is roughly 0
        
        # We want to pull corners UP and OUT.
        # But `openMouth` moves things DOWN (jaw).
        # We need to ignore the jaw movement if possible, or override it.
        # Let's start from NEUTRAL vertices and apply our own delta.
        
        # Define a "Smile Influence" factor.
        # Higher influence at the corners of the mouth (high dist_x).
        # Lower influence in the middle.
        
        # How to detecting Lip vs Jaw?
        # Jaw is lower Y, Lip is higher Y within the moved region.
        # Let's split by median Y of the moved region.
        median_y = np.median(moved_coords[:, 1])
        
        # Only affect Upper Jaw and corners? Or lower lip too?
        # A smile moves corners up.
        
        # Let's select vertices that are "Lips". 
        # Heuristic: Vertices that moved in `openMouth` are mostly mouth related.
        # The top half are upper lip, bottom half lower lip/jaw.
        
        if y > median_y - 0.05: # Upper lip and corners
             # Widen: x = x * (1 + factor)
             # Lift: y = y + factor
             
             # Influence based on X distance (corners move more)
             # Normalize dist_x with respect to max width
             width_x = max(abs(min_bounds[0] - center_bounds[0]), abs(max_bounds[0] - center_bounds[0]))
             if width_x == 0: width_x = 1
             
             rel_x = dist_x / width_x
             
             # Falloff function: x^2 to imply more movement at cornres
             influence = rel_x * rel_x
             
             # Apply deformation
             # 1. Lift corners
             lift_amount = 0.1 * influence
             
             # 2. Widen corners
             widen_factor = 1.0 + (0.2 * influence)
             
             new_x = x * widen_factor
             new_y = y + lift_amount
             
             # Also pull back (Z) slightly for cheek puff?
             new_z = z - (0.05 * influence)
             
             # Blend based on how much it was part of the mutable region
             new_vertices[i] = [new_x, new_y, new_z]
             
        else:
             # Lower lip - just follow slightly?
             # Smile usually stretches lower lip too but maybe not lift as much
             width_x = max(abs(min_bounds[0] - center_bounds[0]), abs(max_bounds[0] - center_bounds[0]))
             if width_x == 0: width_x = 1
             rel_x = dist_x / width_x
             influence = rel_x * rel_x
             
             new_x = x * (1.0 + (0.1 * influence)) # Widen less
             new_y = y + (0.05 * influence) # Lift slightly
             
             new_vertices[i] = [new_x, new_y, z]

    # --- ADDING CHEEK/EYE INFLUENCE ---
    # We need to find vertices that are NOT in the moved_indices (which was mostly mouth/jaw)
    # but are in the cheek region.
    # Cheek Region approx: X between mouth corner X and outer face X. Y above mouth but below eyes.
    
    # Let's Scan ALL vertices for Cheeks
    mouth_top_y = max_bounds[1]
    face_top_y = np.max(neutral_mesh.vertices[:, 1])
    eye_y_approx = mouth_top_y + (face_top_y - mouth_top_y) * 0.4 # Approx eye level
    
    cheek_region_y_min = mouth_top_y
    cheek_region_y_max = eye_y_approx
    
    # X range: Outside of nose (center)
    nose_width = (max_bounds[0] - min_bounds[0]) * 0.3
    
    for i, v in enumerate(neutral_mesh.vertices):
        # Skip if already moved by mouth logic
        if i in moved_indices:
            continue
            
        x, y, z = v
        
        # Check if in Cheek Region
        if cheek_region_y_min < y < cheek_region_y_max:
            if abs(x) > nose_width: # Outside nose
                # This is a Cheek Candidate
                
                # Distance factor
                # Closer to mouth corners = more movement
                dist_from_mouth_center_y = abs(y - center_bounds[1])
                # We want proximity to mouth corners in X too, but simple Y falloff is okay for now
                
                # Check normalized height in cheek region (0 = bottom/mouth, 1 = top/eye)
                norm_h = (y - cheek_region_y_min) / (cheek_region_y_max - cheek_region_y_min)
                
                # Bottom of cheek (near mouth) moves UP more. Top moves less.
                cheek_lift = (1.0 - norm_h) * 0.05
                
                # Puff out (Z)
                # Middle of cheek puffs most
                cheek_puff = np.sin(norm_h * np.pi) * 0.03
                
                new_y = y + cheek_lift
                new_z = z + cheek_puff 
                
                new_vertices[i] = [x, new_y, new_z]


    # Save
    with open(output_path, 'w') as f:
        f.write("# Improved Smile OBJ\n")
        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # Write Original Normals (Good enough approx)
        for n in neutral_mesh.normals:
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
        # Write Faces
        for face in neutral_mesh.faces:
             f_str = " ".join([f"{idx+1}//{idx+1}" for idx in face])
             f.write(f"f {f_str}\n")
             
    print(f"Saved {output_path}")

if __name__ == "__main__":
    # We use "smile.obj" as the 'open mouth' reference since that's what we copied earlier
    # But strictly speaking we should use the one known to mean 'open mouth' to identify the region
    # Current 'smile.obj' IS 'openMouth.obj' copy.
    generate_smile_mesh("neutral.obj", "smile.obj", "smile_improved.obj")

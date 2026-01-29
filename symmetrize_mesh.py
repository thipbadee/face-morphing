
import numpy as np
import sys
import os

# Import our loader
from obj_loader import load_obj, calculate_deltas

def symmetrize_morph(base_path, target_path, output_path, tolerance=0.001):
    print(f"Loading {base_path} and {target_path}...")
    base_mesh = load_obj(base_path)
    target_mesh = load_obj(target_path)
    
    if not base_mesh or not target_mesh:
        return
        
    delta_pos, _ = calculate_deltas(base_mesh, target_mesh)
    
    # Analyze Asymmetry
    left_move = 0
    right_move = 0
    
    # Assume X-axis symmetry. Left = +X, Right = -X (subject to coordinate system)
    # Let's count significant movements (> 0.001)
    
    for i, d in enumerate(delta_pos):
        if np.linalg.norm(d) > 0.001:
            x = base_mesh.vertices[i][0]
            if x > 0:
                left_move += 1
            else:
                right_move += 1
                
    print(f"Vertices moved on +X side: {left_move}")
    print(f"Vertices moved on -X side: {right_move}")
    
    if abs(left_move - right_move) < 10:
        print("Mesh seems relatively symmetric already.")
    else:
        print("Mesh is significantly asymmetric. Attempting to symmetrize...")

    # Symmetrization Logic
    # For every vertex i, finding its mirror pair j such that pos[j] is approx (-pos[i].x, pos[i].y, pos[i].z)
    
    # Build a spatial lookup for base mesh
    # Key: (round(-x, 3), round(y, 3), round(z, 3)) -> index
    # We use tolerance rounding
    
    lookup = {}
    for i, v in enumerate(base_mesh.vertices):
        # Store using quantized coordinates for approximate matching
        key = (int(v[0] / tolerance), int(v[1] / tolerance), int(v[2] / tolerance))
        lookup[key] = i
        
    new_vertices = target_mesh.vertices.copy()
    mirrored_count = 0
    
    for i, v in enumerate(base_mesh.vertices):
        # Look for mirror using -x
        inv_key = (int(-v[0] / tolerance), int(v[1] / tolerance), int(v[2] / tolerance))
        
        mirror_idx = lookup.get(inv_key)
        
        if mirror_idx is not None:
            # We have a pair (i, mirror_idx)
            # If i has delta, apply mirrored delta to mirror_idx
            # But wait, which side is the "good" side?
            # User said "one eyebrow", we assume the moving side is the good one.
            # We'll take the MAX magnitude delta of the pair and apply it symmetrically (mirrored)
            
            d_i = delta_pos[i]
            d_m = delta_pos[mirror_idx]
            
            mag_i = np.linalg.norm(d_i)
            mag_m = np.linalg.norm(d_m)
            
            if mag_i > mag_m and mag_i > 0.001:
                # Mirror i to m
                # Delta m should be (-dx, dy, dz) of Delta i
                mirror_delta = np.array([-d_i[0], d_i[1], d_i[2]])
                # Target m = Base m + Mirror Delta
                new_vertices[mirror_idx] = base_mesh.vertices[mirror_idx] + mirror_delta
                mirrored_count += 1
            elif mag_m > mag_i and mag_m > 0.001:
                # Mirror m to i
                mirror_delta = np.array([-d_m[0], d_m[1], d_m[2]])
                new_vertices[i] = base_mesh.vertices[i] + mirror_delta
                mirrored_count += 1
                
    print(f"Symmetrized {mirrored_count} vertices.")
    
    # Save output
    with open(output_path, 'w') as f:
        f.write(f"# Symmetrized {target_path}\n")
        
        # Vertices
        for v in new_vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
        # Normals (Just copy target's for now, ideally recompute or symmetrize too)
        # But wait, if we changed geometry, normals should change. 
        # For blend shapes, we usually tolerate slight normal errors or recompute.
        # Let's just copy original normals for simplicity, or zero define (obj loader handles missing lines)
        # Better: Copy original normals but try to mirror them too? 
        # Let's write original normals. The loader will load them. 
        # The visual artifact of wrong normal on delta might be small.
        for n in target_mesh.normals:
             f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
             
        # Faces (Copy from base/target)
        # Note: Our loader triangulates, but here we are writing OBJ.
        # We should write the original Faces if possible. 
        # But we parsed them into integer arrays.
        # The loader logic lost the original quad structure.
        # For safety, let's just write TRIANGLES that we have in the mesh object.
        # But wait, our Mesh object from load_obj has `faces` as list of indices.
        # If the loader triangulated, they are triangles.
        for face in base_mesh.faces:
            # OBJ is 1-based
            f_str = " ".join([f"{idx+1}//{idx+1}" for idx in face])
            f.write(f"f {f_str}\n")
            
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    symmetrize_morph("neutral.obj", "surprised.obj", "surprised_symmetric.obj")

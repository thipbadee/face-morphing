
import numpy as np

class Mesh:
    def __init__(self, vertices, normals, faces):
        self.vertices = vertices
        self.normals = normals
        self.faces = faces

def load_obj(filename):
    vertices = []
    normals = []
    faces = []
    
    print(f"Loading {filename}...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                
                if values[0] == 'v':
                    vertices.append([float(x) for x in values[1:4]])
                elif values[0] == 'vn':
                    normals.append([float(x) for x in values[1:4]])
                elif values[0] == 'f':
                    face_indices = []
                    for v in values[1:]:
                        w = v.split('/')
                        face_indices.append(int(w[0]) - 1)
                    
                    # Triangulate (Fan method relative to first vertex)
                    # Works for convex polygons like quads
                    for i in range(1, len(face_indices) - 1):
                        faces.append([face_indices[0], face_indices[i], face_indices[i+1]])
                    
        # Compute Smooth Normals based on topology
        # This ensures the face looks "smooth" regardless of how VN are stored in the file
        vertices_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32)
        
        # Initialize normals
        normals_array = np.zeros_like(vertices_array)
        
        # Accumulate face normals to vertices
        for face in faces_array:
            v0 = vertices_array[face[0]]
            v1 = vertices_array[face[1]]
            v2 = vertices_array[face[2]]
            
            # Cross product for face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Weighted by area? Simple accumulation is usually fine for similar sized polys
            # Normalize face normal first to weight by angle roughly? No, length of cross product is 2*Area. 
            # Weighted by Area is good.
            
            normals_array[face[0]] += face_normal
            normals_array[face[1]] += face_normal
            normals_array[face[2]] += face_normal
            
        # Normalize result
        # Avoid division by zero
        norms = np.linalg.norm(normals_array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals_array = normals_array / norms
                    
        return Mesh(vertices_array, 
                   normals_array, 
                   faces_array)
                   
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def validate_topology(base_mesh, target_mesh):
    """
    Checks if two meshes have the same number of vertices and faces.
    Required for blend shapes.
    """
    if base_mesh is None or target_mesh is None:
        return False
        
    if len(base_mesh.vertices) != len(target_mesh.vertices):
        raise ValueError(f"Topology Mismatch: Vertex counts do not match. Base: {len(base_mesh.vertices)}, Target: {len(target_mesh.vertices)}")
        return False
        
    # In a real rigorous check, we might check face connectivity too, 
    # but vertex count is the critical one for VBOs of same length.
    print("Topology Check Passed: Vertex counts match.")
    return True

def calculate_deltas(base_mesh, target_mesh):
    """
    Calculates the difference vectors (deltas) between target and base.
    """
    # Verify we have normals; if not, fill with zeros or re-calculate (simplified here)
    if len(base_mesh.normals) == 0:
        base_mesh.normals = np.zeros_like(base_mesh.vertices)
    if len(target_mesh.normals) == 0:
        target_mesh.normals = np.zeros_like(target_mesh.vertices)
        
    if len(base_mesh.normals) != len(target_mesh.normals):
         # If normals don't match, just use 0 deltas for normals to avoid crash, but warn
         print("Warning: Normal counts mismatch or missing. Lighting may be incorrect.")
         # Resize or fill logic could go here, for now assume strict matching or fill 0
         if len(target_mesh.normals) == 0:
             target_mesh.normals = np.zeros_like(base_mesh.normals)

    delta_pos = target_mesh.vertices - base_mesh.vertices
    delta_norm = target_mesh.normals - base_mesh.normals
    
    return delta_pos, delta_norm

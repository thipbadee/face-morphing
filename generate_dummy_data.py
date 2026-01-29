
import os
import math

def create_sphere_face(filename, is_smile=False):
    vertices = []
    normals = []
    faces = []
    
    radius = 1.0
    lat_bands = 20
    long_bands = 20
    
    # Generate Vertices and Normals
    for lat in range(lat_bands + 1):
        theta = lat * math.pi / lat_bands
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        for lon in range(long_bands + 1):
            phi = lon * 2 * math.pi / long_bands
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            
            # Deformation for "Face" features
            # Let's say -Z is the face direction (standard OpenGL camera looks down -Z)
            # Actually our camera is at +Z looking at -Z (camera_front = -1). 
            # So the face should look towards +Z to be seen by camera at +Z? 
            # Wait, camera is at (0,0,3) looking at (0,0,0). 
            # Objects at (0,0,0).
            # If object looks at +Z, it looks at camera.
            
            # Base shape adjustment (Head shape)
            # Flatten slightly in Z
            z *= 0.9 
            
            # Smile Deformation
            if is_smile:
                # Identify "mouth" area: Lower middle, front side (+Z)
                # Front is +Z (sin_phi > 0 depending on rotation, let's assume standard UV sphere mapping)
                # x = cos(phi)sin(theta), y = cos(theta), z = sin(phi)sin(theta)
                # Let's assume +Z is front. 
                
                # Check if vertex is in the "mouth" region
                # Approx: y between -0.5 and -0.2, z > 0.5 (front), x between -0.5 and 0.5
                if z > 0.4 and -0.6 < y < -0.2:
                    # Pull corners of mouth up and out
                    dist_from_center_x = abs(x)
                    # Simple smile: lift y based on x distance
                    if dist_from_center_x > 0.1:
                        y += 0.1 * dist_from_center_x
                        x *= 1.1 # Widen
            
            vertices.append([x * radius, y * radius, z * radius])
            normals.append([x, y, z]) # Normalized sphere normals are just the position vector
            
    # Generate Faces
    # v indices are 1-based in OBJ
    for lat in range(lat_bands):
        for lon in range(long_bands):
            first = (lat * (long_bands + 1)) + lon + 1
            second = first + long_bands + 1
            
            # Two triangles per quad
            # 1---2
            # | / |
            # 3---4
            
            # v1, v2, v3
            faces.append([first, second, first + 1])
            # v2, v4, v3
            faces.append([second, second + 1, first + 1])
            
    with open(filename, 'w') as f:
        f.write("# Dummy Sphere Face OBJ\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            
        for n in normals:
            f.write(f"vn {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n")
            
        for face in faces:
            f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[2]}//{face[2]}\n")

if __name__ == "__main__":
    print("Generating neutral.obj (Sphere Head)...")
    create_sphere_face("neutral.obj", is_smile=False)
    
    print("Generating smile.obj (Sphere Head with Smile)...")
    create_sphere_face("smile.obj", is_smile=True)
    print("Done.")

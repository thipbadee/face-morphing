
import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes
import os
import sys

# Add current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from obj_loader import load_obj, validate_topology, calculate_deltas

# --- Shader Source Code ---
VERTEX_SHADER_SOURCE = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aDeltaPos1;
layout(location = 3) in vec3 aDeltaNorm1;
layout(location = 4) in vec3 aDeltaPos2;
layout(location = 5) in vec3 aDeltaNorm2;

uniform float weight1; // Smile
uniform float weight2; // Surprise
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 FragPos;
out vec3 Normal;

void main() {
    vec3 finalPos = aPos + (aDeltaPos1 * weight1) + (aDeltaPos2 * weight2);
    vec3 finalNormal = aNormal + (aDeltaNorm1 * weight1) + (aDeltaNorm2 * weight2); 
    
    gl_Position = projection * view * model * vec4(finalPos, 1.0);
    
    FragPos = vec3(model * vec4(finalPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * finalNormal; 
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main() {
    // Ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
"""

# Global variables
morph_weight1 = 0.0 # Smile
morph_weight2 = 0.0 # Surprise
last_x = 400
last_y = 300
first_mouse = True
yaw = -90.0
pitch = 0.0
camera_pos = glm.vec3(0.0, 0.0, 3.0)
camera_front = glm.vec3(0.0, 0.0, -1.0)
camera_up = glm.vec3(0.0, 1.0, 0.0)

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program():
    vertex_shader = compile_shader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def process_input(window):
    global morph_weight1, morph_weight2
    
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    
    # Morph Control 1 (Smile)
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        morph_weight1 += 0.01
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        morph_weight1 -= 0.01
        
    # Morph Control 2 (Surprise)
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        morph_weight2 += 0.01
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        morph_weight2 -= 0.01
        
    # Clamp weights
    morph_weight1 = max(0.0, min(1.0, morph_weight1))
    morph_weight2 = max(0.0, min(1.0, morph_weight2))

def main():
    # Initialize GLFW
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Face Morphing Demo", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Compile shaders
    try:
        shader_program = create_shader_program()
    except Exception as e:
        print(f"Shader compilation failed: {e}")
        glfw.terminate()
        return

    # Load Models
    print("Loading models...")
    base_mesh = load_obj("neutral.obj")
    target_mesh1 = load_obj("smile_improved.obj")
    target_mesh2 = load_obj("surprised_symmetric.obj")
    
    if not base_mesh or not target_mesh1 or not target_mesh2:
        print("Failed to load models. Please ensure neutral.obj, smile_improved.obj and surprised.obj exist.")
        glfw.terminate()
        return

    # Validate
    try:
        validate_topology(base_mesh, target_mesh1)
        validate_topology(base_mesh, target_mesh2)
    except ValueError as e:
        print(f"Validation Error: {e}")
        glfw.terminate()
        return

    # Calculate Bounding Box and Center
    min_vals = np.min(base_mesh.vertices, axis=0)
    max_vals = np.max(base_mesh.vertices, axis=0)
    center = (min_vals + max_vals) / 2.0
    size = max_vals - min_vals
    max_dim = np.max(size)
    
    print(f"Mesh Bounds: Min={min_vals}, Max={max_vals}")
    print(f"Center: {center}")
    print(f"Size: {size}")

    # Calculate Deltas
    delta_pos1, delta_norm1 = calculate_deltas(base_mesh, target_mesh1)
    delta_pos2, delta_norm2 = calculate_deltas(base_mesh, target_mesh2)


    # Flatten arrays for OpenGL
    # Structure: [x,y,z, nx,ny,nz, dx1,dy1,dz1, dnx1,dny1,dnz1, dx2,dy2,dz2, dnx2,dny2,dnz2]
    
    num_vertices = len(base_mesh.vertices)
    
    vertex_data = []
    for i in range(num_vertices):
        # Position
        vertex_data.extend(base_mesh.vertices[i])
        # Normal
        if i < len(base_mesh.normals):
            vertex_data.extend(base_mesh.normals[i])
        else:
            vertex_data.extend([0,0,0])
            
        # Delta 1
        vertex_data.extend(delta_pos1[i])
        vertex_data.extend(delta_norm1[i])
        
        # Delta 2
        vertex_data.extend(delta_pos2[i])
        vertex_data.extend(delta_norm2[i])
        
    vertex_data = np.array(vertex_data, dtype=np.float32)

    # Create Index Buffer (EBO)
    # Flatten faces
    indices = []
    for face in base_mesh.faces:
        indices.extend(face)
    indices = np.array(indices, dtype=np.uint32)

    # Setup Bumpers
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    EBO = glGenBuffers(1)

    glBindVertexArray(VAO)

    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # Vertex Attributes
    # Stride = 3 + 3 + 3 + 3 + 3 + 3 floats = 18 floats * 4 bytes
    stride = 18 * 4 
    
    # 0: Pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    # 1: Normal
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    
    # 2: Delta Pos 1
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    glEnableVertexAttribArray(2)
    
    # 3: Delta Norm 1
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(9 * 4))
    glEnableVertexAttribArray(3)
    
    # 4: Delta Pos 2
    glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12 * 4))
    glEnableVertexAttribArray(4)
    
    # 5: Delta Norm 2
    glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(15 * 4))
    glEnableVertexAttribArray(5)

    # Render Loop
    glEnable(GL_DEPTH_TEST)

    while not glfw.window_should_close(window):
        # Input
        process_input(window)

        # Clear
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Activate Shader
        glUseProgram(shader_program)

        # Update Uniforms
        glUniform1f(glGetUniformLocation(shader_program, "weight1"), morph_weight1)
        glUniform1f(glGetUniformLocation(shader_program, "weight2"), morph_weight2)
        
        # Matrices
        model = glm.mat4(1.0)
        # 1. Translate to center the object at origin
        model = glm.translate(model, glm.vec3(-center[0], -center[1], -center[2])) 
        # 2. Scale to fit nicely (optional, but good for normalization)
        scale_factor = 2.0 / max_dim # Scale to roughly size 2.0
        model = glm.scale(model, glm.vec3(scale_factor))
        
        view = glm.lookAt(glm.vec3(0.0, 0.0, 3.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
        projection = glm.perspective(glm.radians(45.0), 800/600, 0.1, 100.0)

        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
        
        # Lighting Uniforms
        glUniform3f(glGetUniformLocation(shader_program, "lightPos"), 2.0, 2.0, 2.0)
        glUniform3f(glGetUniformLocation(shader_program, "viewPos"), camera_pos.x, camera_pos.y, camera_pos.z)
        glUniform3f(glGetUniformLocation(shader_program, "lightColor"), 1.0, 1.0, 1.0)
        glUniform3f(glGetUniformLocation(shader_program, "objectColor"), 1.0, 0.5, 0.31) # Coral color

        # Draw
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # Title Update
        glfw.set_window_title(window, f"Face Morphing Demo - Smile: {morph_weight1:.2f}, Surprise: {morph_weight2:.2f}")

    glfw.terminate()


if __name__ == "__main__":
    main()

# Face Morphing Demo

This project demonstrates 3D face morphing using OpenGL and Python. It interpolates between a neutral face and target facial expressions (smile, surprise) using blend shapes (linear interpolation of vertex positions and normals).

## Features

- **3D Face Rendering:** Loads and renders 3D OBJ models.
- **Real-time Morphing:** interactive control over morph weights to blend between expressions.
- **Lighting:** smooth shading with ambient, diffuse, and specular lighting (Phong model).
- **Camera Control:** Basic camera setup.

## Requirements

- Python 3.x
- `glfw`
- `PyOpenGL`
- `numpy`
- `glm`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/thipbadee/face-morphing.git
    cd face-morphing
    ```

2.  Install dependencies:
    ```bash
    pip install glfw PyOpenGL numpy PyGLM
    ```

## Usage

1.  Ensure you have the model files in the project directory:
    - `neutral.obj` (Base mesh)
    - `angry.obj` (Target mesh 1)
    - `surprise.obj` (Target mesh 2)

2.  Run the main script:
    ```bash
    python main.py
    ```

3.  Controls:
    - **Right Arrow / Left Arrow**: Increase/Decrease morph weight for Expression 1 (Angry/Smile).
    - **Up Arrow / Down Arrow**: Increase/Decrease morph weight for Expression 2 (Surprise).
    - **ESC**: Exit the application.

## Project Structure

- `main.py`: Main entry point, handles window creation, input, rendering loop, and shader management.
- `obj_loader.py`: Utility functions to load OBJ files, validate topology, and calculate deltas for blend shapes.
- `.gitignore`: Specifies intentionally untracked files to ignore.

import os
import subprocess


def compile_shader(shader_path):
    shader_dir = os.path.dirname(shader_path)
    shader_name, shader_ext = os.path.splitext(os.path.basename(shader_path))

    output_path = os.path.join(shader_dir, "compiled", f"{shader_name}.spv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    command = f"glslc -o {output_path} {shader_path}"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Compiled {shader_path} to SPIR-V successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile {shader_path}. Error: {e}")


def compile_all_shaders(shader_dir):
    shader_extensions = [".comp", ".vert", ".frag"]

    for root, dirs, files in os.walk(shader_dir):
        for file in files:
            if any(file.endswith(ext) for ext in shader_extensions):
                file_path = os.path.join(root, file)
                compile_shader(file_path)


if __name__ == "__main__":
    shaders_folder = "./res/shaders"
    compile_all_shaders(shaders_folder)

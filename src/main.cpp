#include <print>

#include <ggml/ggml.h>
#include <stable-diffusion.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include <stb_image_write.h>

// example usage only
constexpr auto model_path = "example.safetensors";

int main() {
    StableDiffusion sd;
    if (!sd.load_from_file(
            model_path,
            "", GGML_TYPE_F16))
        return 1;

    auto res = sd.txt2img("the 2024 new year", "", 7.0f, 512, 512, EULER_A, 20, 69, 1);

    stbi_write_png("output.png", 512, 512, 3, res[0], 0);

    std::print("done.");
    return 0;
}

#include "app/application.hpp"

int main(int argc, char* argv[]) {
  // TODO: this changes the current working dir, for you it needs to be `<repo-location>/cmake-build-debug`
  std::print("Please check the cwd from below in code!");
  std::filesystem::current_path("D:\\diffusion-studio\\cmake-build-debug");

  std::vector<std::string_view> args(argv, argv + argc);
  auto app = std::make_unique<DiffusionStudio::Application>(args);

  return app->Run();
}

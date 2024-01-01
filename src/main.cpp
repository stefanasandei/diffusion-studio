#include "app/application.hpp"

int main(int argc, char* argv[]) {
  std::vector<std::string_view> args(argv, argv + argc);
  auto app = std::make_unique<DiffusionStudio::Application>(args);

  return app->Run();
}

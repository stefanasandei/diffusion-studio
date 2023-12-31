//
// Created by Stefan on 12/24/2023.
//

#include "platform/window.h"

namespace platform {

Window::Window(glm::ivec2 size, const std::string &title) {
  util::error::ErrNDie(!glfwInit(), "Failed to init glfw.");

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  m_Window = glfwCreateWindow(size.x, size.y, title.c_str(), nullptr, nullptr);
  util::error::ErrNDie(!m_Window, "Failed to create a window");

  glfwMakeContextCurrent(m_Window);
}

Window::~Window() { glfwTerminate(); }

void Window::PollEvents() const {
  glfwSwapBuffers(m_Window);
  glfwPollEvents();
}

bool Window::ShouldClose() const { return glfwWindowShouldClose(m_Window); }

GLFWwindow *Window::GetNative() { return m_Window; }

glm::ivec2 Window::GetSize() {
  int32_t width, height;
  glfwGetWindowSize(m_Window, &width, &height);
  return {width, height};
}

}  // namespace platform

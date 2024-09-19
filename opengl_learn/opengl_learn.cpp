#include <iostream>

#include <glad/glad.h>
// include glad.h first
#include <GLFW/glfw3.h>

int main() {
    // 初始化GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    // 创建窗口
    GLFWwindow* window = glfwCreateWindow(800, 600, "GLFW Window", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    // 将窗口的上下文设置为当前线程的主上下文
    glfwMakeContextCurrent(window);

    // 初始化GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD!" << std::endl;
        return -1;
    }

    // 设置视口大小
    glViewport(0, 0, 800, 600);

    // 设置窗口大小调整的回调
    glfwSetFramebufferSizeCallback(window,
                                   [](GLFWwindow* window, int width, int height) { glViewport(0, 0, width, height); });

    // 主循环
    while (!glfwWindowShouldClose(window)) {
        // 清空颜色缓冲
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 处理GLFW事件
        glfwPollEvents();

        // 交换缓冲
        glfwSwapBuffers(window);
    }

    // 终止GLFW
    glfwTerminate();
    return 0;
}

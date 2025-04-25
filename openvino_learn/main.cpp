#include "openvino_utils.hpp"

int main(int argc, char const* argv[]) {
    fmt::println("OpenVINO Version: {}", ov::get_openvino_version());
    auto model = Ahri::OpenVINO::OpenVINOModel{"/media/supervisor/windowsd/Code/github/Vision/models/VGG.xml"};
    model.inference();
    return 0;
}

#include <filesystem>

#include "Vision.hpp"

#include "openvino_utils.hpp"

int main(int argc, char const* argv[]) {
    fmt::println("OpenVINO Version: {}", ov::get_openvino_version());
    auto model_path = std::filesystem::path{VISION_ROOT} / "models/VGG.xml";
    auto model = Ahri::OpenVINO::OpenVINOModel{model_path};
    model.inference();
    return 0;
}

/**
 * @file ffmpeg_learn.cpp
 * @date 2024/09/10
 * @author Sokyoei
 *
 *
 */

#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
}

namespace Ahri {}  // namespace Ahri

int main(int argc, char const* argv[]) {
    std::cout << "avcodec config: " << avcodec_configuration() << '\n';
    return 0;
}

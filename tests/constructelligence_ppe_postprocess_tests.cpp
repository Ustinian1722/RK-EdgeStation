#include "constructelligence_ppe_postprocess.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    constexpr int channels = 18;
    constexpr int anchors = 5;
    std::array<float, channels * anchors> output{};
    const auto set_box = [&](int anchor, int source_class, float score,
                             float cx, float cy, float width, float height) {
        output[anchor] = cx;
        output[anchors + anchor] = cy;
        output[2 * anchors + anchor] = width;
        output[3 * anchors + anchor] = height;
        output[(4 + source_class) * anchors + anchor] = score;
    };
    set_box(0, 0, 0.90F, 320.0F, 320.0F, 100.0F, 100.0F);  // Person -> 5
    set_box(1, 0, 0.80F, 322.0F, 321.0F, 100.0F, 100.0F);  // duplicate person
    set_box(2, 1, 0.75F, 320.0F, 320.0F, 80.0F, 60.0F);   // hardhat -> 0
    set_box(3, 4, 0.70F, 500.0F, 300.0F, 40.0F, 40.0F);   // no vest -> 11
    set_box(4, 8, 0.99F, 150.0F, 150.0F, 40.0F, 40.0F);  // unsupported excavator

    letterbox_t letterbox{};
    letterbox.x_pad = 0;
    letterbox.y_pad = 80;
    letterbox.scale = 1.0F;
    const auto detections = edge::decode_constructelligence_ppe(
        output.data(), channels, anchors, letterbox, 640, 480, 0.25F, 0.45F);
    assert(detections.size() == 3);
    assert(detections[0].class_id == 5);
    assert(detections[1].class_id == 0);
    assert(detections[2].class_id == 11);
    assert(detections[0].top == 190 && detections[0].bottom == 290);
    assert(std::abs(detections[0].confidence - 0.90F) < 1e-6F);
    return 0;
}

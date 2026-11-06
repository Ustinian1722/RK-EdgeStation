#include "preprocess.hpp"
#include "yolox_ppe_postprocess.hpp"

#include <opencv2/imgcodecs.hpp>

#include <rknn_api.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct RknnSession {
    rknn_context handle = 0;
    rknn_app_context_t app{};
    std::vector<rknn_tensor_attr> inputs;
    std::vector<rknn_tensor_attr> outputs;

    ~RknnSession() {
        if (handle != 0) rknn_destroy(handle);
    }
};

struct OutputsGuard {
    rknn_context context;
    std::uint32_t count;
    rknn_output* outputs;
    ~OutputsGuard() { rknn_outputs_release(context, count, outputs); }
};

void require_rknn(int status, const char* operation) {
    if (status != RKNN_SUCC) {
        throw std::runtime_error(std::string(operation) + " failed with code " +
                                 std::to_string(status));
    }
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) escaped += '?';
                else escaped += static_cast<char>(ch);
        }
    }
    return escaped;
}

void open_model(RknnSession& session, const std::string& path) {
    std::ifstream model_file(path, std::ios::binary | std::ios::ate);
    if (!model_file) throw std::runtime_error("could not open RKNN model: " + path);
    const auto end = model_file.tellg();
    if (end <= 0) throw std::runtime_error("RKNN model is empty: " + path);
    std::vector<std::uint8_t> model(static_cast<std::size_t>(end));
    model_file.seekg(0, std::ios::beg);
    if (!model_file.read(reinterpret_cast<char*>(model.data()),
                         static_cast<std::streamsize>(model.size()))) {
        throw std::runtime_error("could not read RKNN model: " + path);
    }

    require_rknn(rknn_init(&session.handle, model.data(),
                           static_cast<std::uint32_t>(model.size()), 0, nullptr), "rknn_init");
    session.app.rknn_ctx = session.handle;
    require_rknn(rknn_query(session.handle, RKNN_QUERY_IN_OUT_NUM,
                            &session.app.io_num, sizeof(session.app.io_num)), "query I/O count");
    if (session.app.io_num.n_input != 1 || session.app.io_num.n_output != 3) {
        throw std::runtime_error("PPE evaluator requires one input and three YOLOX output maps");
    }

    session.inputs.resize(session.app.io_num.n_input);
    session.outputs.resize(session.app.io_num.n_output);
    for (std::uint32_t index = 0; index < session.app.io_num.n_input; ++index) {
        session.inputs[index] = {};
        session.inputs[index].index = index;
        require_rknn(rknn_query(session.handle, RKNN_QUERY_INPUT_ATTR,
                                &session.inputs[index], sizeof(rknn_tensor_attr)),
                     "query input tensor");
    }
    for (std::uint32_t index = 0; index < session.app.io_num.n_output; ++index) {
        session.outputs[index] = {};
        session.outputs[index].index = index;
        require_rknn(rknn_query(session.handle, RKNN_QUERY_OUTPUT_ATTR,
                                &session.outputs[index], sizeof(rknn_tensor_attr)),
                     "query output tensor");
        const auto& output = session.outputs[index];
        const bool channels_first = output.n_dims == 4 && output.dims[0] == 1 &&
                                    output.dims[1] == 17 && output.fmt == RKNN_TENSOR_NCHW;
        const bool channels_last = output.n_dims == 4 && output.dims[0] == 1 &&
                                   output.dims[3] == 17 && output.fmt == RKNN_TENSOR_NHWC;
        if (!channels_first && !channels_last) {
            throw std::runtime_error("RKNN output is not a supported 17-channel YOLOX tensor");
        }
    }

    session.app.input_attrs = session.inputs.data();
    session.app.output_attrs = session.outputs.data();
    const auto& input = session.inputs.front();
    if (input.n_dims != 4 || input.fmt != RKNN_TENSOR_NHWC || input.dims[3] != 3) {
        throw std::runtime_error("PPE evaluator requires an NHWC three-channel input");
    }
    session.app.model_height = static_cast<int>(input.dims[1]);
    session.app.model_width = static_cast<int>(input.dims[2]);
    session.app.model_channel = static_cast<int>(input.dims[3]);
    session.app.is_quant = session.outputs.front().type == RKNN_TENSOR_INT8 &&
                           session.outputs.front().qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;
    std::cerr << "model=" << path << " input=" << session.app.model_width << 'x'
              << session.app.model_height << " input_type=" << input.type
              << " quantized_outputs=" << (session.app.is_quant ? "true" : "false") << '\n';
}

void infer_one(RknnSession& session, const cv::Mat& bgr, float threshold,
               object_detect_result_list& detections) {
    edge::DecodedFrame frame;
    frame.width = bgr.cols;
    frame.height = bgr.rows;
    frame.bgr = bgr;
    auto preprocessed = edge::preprocess_frame(frame, session.app.model_width,
                                               session.app.model_height, false,
                                               edge::PreprocessBackend::Cpu);
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<std::uint32_t>(preprocessed.model_input.total() *
                                            preprocessed.model_input.elemSize());
    input.buf = preprocessed.model_input.data;
    require_rknn(rknn_inputs_set(session.handle, 1, &input), "rknn_inputs_set");
    require_rknn(rknn_run(session.handle, nullptr), "rknn_run");

    std::vector<rknn_output> outputs(session.app.io_num.n_output);
    for (std::uint32_t index = 0; index < session.app.io_num.n_output; ++index) {
        outputs[index] = {};
        outputs[index].index = index;
        outputs[index].want_float = session.app.is_quant ? 0 : 1;
    }
    require_rknn(rknn_outputs_get(session.handle, session.app.io_num.n_output,
                                  outputs.data(), nullptr), "rknn_outputs_get");
    OutputsGuard guard{session.handle, session.app.io_num.n_output, outputs.data()};
    const int status = post_process_yolox_ppe(&session.app, outputs.data(),
        &preprocessed.letterbox, threshold, 0.45F, bgr.cols, bgr.rows, &detections);
    if (status != 0) {
        throw std::runtime_error("PPE postprocess failed with code " + std::to_string(status));
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7 || std::string(argv[1]) != "--model" ||
        std::string(argv[3]) != "--image-manifest" || std::string(argv[5]) != "--output") {
        std::cerr << "usage: ppe_rknn_dataset_eval --model MODEL.rknn "
                     "--image-manifest IMAGE_ROOT_AND_FILES.txt --output OUTPUT.jsonl\n"
                  << "manifest format: first line is image root; remaining lines are filenames\n";
        return 2;
    }
    try {
        const std::string model_path = argv[2];
        const std::string manifest_path = argv[4];
        const std::string output_path = argv[6];
        std::ifstream manifest(manifest_path);
        if (!manifest) throw std::runtime_error("could not open image manifest: " + manifest_path);
        std::string image_root;
        if (!std::getline(manifest, image_root) || image_root.empty()) {
            throw std::runtime_error("image manifest is missing its root directory");
        }
        if (!image_root.empty() && image_root.back() == '\r') image_root.pop_back();
        std::vector<std::string> file_names;
        for (std::string line; std::getline(manifest, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty() && line.front() != '#') file_names.push_back(line);
        }
        if (file_names.empty()) throw std::runtime_error("image manifest contains no filenames");

        std::ofstream output(output_path);
        if (!output) throw std::runtime_error("could not create prediction JSONL");
        output << std::fixed << std::setprecision(6);

        RknnSession session;
        open_model(session, model_path);
        constexpr float kLowConfidenceForAp = 0.001F;
        for (std::size_t index = 0; index < file_names.size(); ++index) {
            const std::string image_path = image_root + "/" + file_names[index];
            const cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
            if (bgr.empty()) throw std::runtime_error("could not decode image: " + image_path);
            object_detect_result_list detections{};
            infer_one(session, bgr, kLowConfidenceForAp, detections);
            output << "{\"index\":" << index << ",\"file_name\":\""
                   << json_escape(file_names[index]) << "\",\"width\":" << bgr.cols
                   << ",\"height\":" << bgr.rows << ",\"detections\":[";
            for (int item = 0; item < detections.count; ++item) {
                if (item != 0) output << ',';
                const auto& detection = detections.results[item];
                output << "{\"class_id\":" << detection.cls_id
                       << ",\"score\":" << detection.prop << ",\"bbox_xyxy\":["
                       << detection.box.left << ',' << detection.box.top << ','
                       << detection.box.right << ',' << detection.box.bottom << "]}";
            }
            output << "]}\n";
        }
        output.flush();
        if (!output) throw std::runtime_error("failed writing prediction JSONL");
        std::cerr << "completed_images=" << file_names.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ppe_rknn_dataset_eval: " << error.what() << '\n';
        return 1;
    }
}

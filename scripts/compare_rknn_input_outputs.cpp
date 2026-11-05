// Compare RKNN outputs for two saved UINT8 NHWC image tensors.
// Build on RK3576: g++ -O2 -std=c++17 scripts/compare_rknn_input_outputs.cpp -o rknn-input-compare -lrknnrt

#include <rknn_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string first_input;
    std::string second_input;
    std::string dump_outputs_dir;
    std::size_t expected_input_bytes = 640U * 640U * 3U;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        if (key == "--help" || key == "-h") {
            std::cout << "compare_rknn_input_outputs --model MODEL.rknn --first INPUT.bin "
                         "--second INPUT.bin [--expected-input-bytes N] [--dump-outputs DIR]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
        const std::string value(argv[++i]);
        if (key == "--model") options.model = value;
        else if (key == "--first") options.first_input = value;
        else if (key == "--second") options.second_input = value;
        else if (key == "--dump-outputs") options.dump_outputs_dir = value;
        else if (key == "--expected-input-bytes") options.expected_input_bytes = std::stoull(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    if (options.model.empty() || options.first_input.empty() || options.second_input.empty()) {
        throw std::runtime_error("--model, --first and --second are required");
    }
    return options;
}

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open file: " + path);
    const auto end = input.tellg();
    if (end <= 0) throw std::runtime_error("empty file: " + path);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input) throw std::runtime_error("failed to read file: " + path);
    return bytes;
}

class RknnContext {
public:
    explicit RknnContext(const std::vector<std::uint8_t>& model) {
        const int status = rknn_init(&context_, const_cast<std::uint8_t*>(model.data()),
                                     static_cast<std::uint32_t>(model.size()), 0, nullptr);
        if (status != RKNN_SUCC) throw std::runtime_error("rknn_init failed: " + std::to_string(status));
    }
    ~RknnContext() {
        if (context_ != 0) rknn_destroy(context_);
    }
    RknnContext(const RknnContext&) = delete;
    RknnContext& operator=(const RknnContext&) = delete;
    rknn_context get() const { return context_; }

private:
    rknn_context context_ = 0;
};

using TensorOutputs = std::vector<std::vector<float>>;

void save_outputs(const TensorOutputs& outputs, const std::vector<rknn_tensor_attr>& attributes,
                  const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream manifest(root / "tensor-manifest.txt");
    if (!manifest) throw std::runtime_error("cannot create output tensor manifest");
    manifest << "input_layout=NHWC\ninput_type=uint8\ninput_shape=640,640,3\n"
             << "source_width=768\nsource_height=576\nletterbox_x_pad=0\n"
             << "letterbox_y_pad=80\nletterbox_scale=0.8333333333\n"
             << "output_requested_type=float32\n";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        std::ostringstream filename;
        filename << "output-" << std::setw(2) << std::setfill('0') << i << ".f32.bin";
        std::ofstream output(root / filename.str(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(outputs[i].data()),
                     static_cast<std::streamsize>(outputs[i].size() * sizeof(float)));
        if (!output) throw std::runtime_error("failed to write RKNN output tensor");
        manifest << "output[" << i << "].name=" << attributes[i].name << '\n'
                 << "output[" << i << "].shape=";
        for (std::uint32_t dim = 0; dim < attributes[i].n_dims; ++dim) {
            if (dim != 0) manifest << ',';
            manifest << attributes[i].dims[dim];
        }
        manifest << '\n' << "output[" << i << "].file=" << filename.str() << '\n';
    }
    manifest.flush();
    if (!manifest) throw std::runtime_error("failed to flush output tensor manifest");
}

TensorOutputs infer(rknn_context context, const std::vector<std::uint8_t>& input_bytes,
                   const rknn_input_output_num& io_num) {
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<std::uint32_t>(input_bytes.size());
    input.buf = const_cast<std::uint8_t*>(input_bytes.data());
    int status = rknn_inputs_set(context, 1, &input);
    if (status != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed: " + std::to_string(status));
    status = rknn_run(context, nullptr);
    if (status != RKNN_SUCC) throw std::runtime_error("rknn_run failed: " + std::to_string(status));

    std::vector<rknn_output> outputs(io_num.n_output);
    for (std::uint32_t i = 0; i < io_num.n_output; ++i) {
        outputs[i] = {};
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    status = rknn_outputs_get(context, io_num.n_output, outputs.data(), nullptr);
    if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed: " + std::to_string(status));

    TensorOutputs copied;
    copied.reserve(outputs.size());
    for (const auto& output : outputs) {
        if (output.buf == nullptr || output.size == 0 || output.size % sizeof(float) != 0) {
            rknn_outputs_release(context, io_num.n_output, outputs.data());
            throw std::runtime_error("RKNN returned an invalid float output");
        }
        const auto* begin = static_cast<const float*>(output.buf);
        copied.emplace_back(begin, begin + output.size / sizeof(float));
    }
    status = rknn_outputs_release(context, io_num.n_output, outputs.data());
    if (status != RKNN_SUCC) throw std::runtime_error("rknn_outputs_release failed: " + std::to_string(status));
    return copied;
}

struct Metrics {
    double mean_abs = 0.0;
    double rmse = 0.0;
    double max_abs = 0.0;
    double cosine = 1.0;
};

Metrics compare(const std::vector<float>& first, const std::vector<float>& second) {
    if (first.size() != second.size()) throw std::runtime_error("RKNN output element count mismatch");
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    double dot = 0.0;
    double norm_first = 0.0;
    double norm_second = 0.0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < first.size(); ++i) {
        const double a = first[i];
        const double b = second[i];
        const double difference = std::abs(a - b);
        sum_abs += difference;
        sum_squared += difference * difference;
        maximum = std::max(maximum, difference);
        dot += a * b;
        norm_first += a * a;
        norm_second += b * b;
    }
    const double count = static_cast<double>(first.size());
    return {sum_abs / count, std::sqrt(sum_squared / count), maximum,
            dot / (std::sqrt(norm_first * norm_second) + 1e-12)};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto model = read_bytes(options.model);
        const auto first = read_bytes(options.first_input);
        const auto second = read_bytes(options.second_input);
        if (first.size() != options.expected_input_bytes || second.size() != options.expected_input_bytes) {
            throw std::runtime_error("input byte count does not match --expected-input-bytes");
        }

        RknnContext context(model);
        rknn_input_output_num io_num{};
        const int status = rknn_query(context.get(), RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (status != RKNN_SUCC || io_num.n_input != 1 || io_num.n_output == 0) {
            throw std::runtime_error("model must have one input and at least one output");
        }
        rknn_tensor_attr input_attr{};
        input_attr.index = 0;
        if (rknn_query(context.get(), RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)) != RKNN_SUCC ||
            input_attr.n_dims != 4 || input_attr.fmt != RKNN_TENSOR_NHWC ||
            input_attr.dims[1] != 640 || input_attr.dims[2] != 640 || input_attr.dims[3] != 3) {
            throw std::runtime_error("model must accept a 640x640x3 NHWC input");
        }
        std::vector<rknn_tensor_attr> output_attributes(io_num.n_output);
        for (std::uint32_t i = 0; i < io_num.n_output; ++i) {
            output_attributes[i] = {};
            output_attributes[i].index = i;
            if (rknn_query(context.get(), RKNN_QUERY_OUTPUT_ATTR, &output_attributes[i],
                           sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
                throw std::runtime_error("failed to query RKNN output attribute");
            }
        }
        const auto first_outputs = infer(context.get(), first, io_num);
        const auto second_outputs = infer(context.get(), second, io_num);
        if (!options.dump_outputs_dir.empty()) {
            const std::filesystem::path root(options.dump_outputs_dir);
            save_outputs(first_outputs, output_attributes, root / "cpu");
            save_outputs(second_outputs, output_attributes, root / "rga");
        }

        std::size_t changed_outputs = 0;
        std::cout << std::fixed << std::setprecision(9)
                  << "{\"type\":\"rknn_input_output_comparison\",\"model_bytes\":" << model.size()
                  << ",\"input_bytes\":" << first.size()
                  << ",\"input_mean_abs_difference\":";
        double input_difference = 0.0;
        for (std::size_t i = 0; i < first.size(); ++i) {
            input_difference += std::abs(static_cast<int>(first[i]) - static_cast<int>(second[i]));
        }
        std::cout << input_difference / static_cast<double>(first.size())
                  << ",\"outputs\":[";
        for (std::size_t i = 0; i < first_outputs.size(); ++i) {
            const Metrics metrics = compare(first_outputs[i], second_outputs[i]);
            if (metrics.max_abs != 0.0) ++changed_outputs;
            if (i != 0) std::cout << ',';
            std::cout << "{\"index\":" << i << ",\"elements\":" << first_outputs[i].size()
                      << ",\"mean_abs_difference\":" << metrics.mean_abs
                      << ",\"rmse\":" << metrics.rmse
                      << ",\"max_abs_difference\":" << metrics.max_abs
                      << ",\"cosine_similarity\":" << metrics.cosine << '}';
        }
        std::cout << "],\"changed_output_count\":" << changed_outputs
                  << ",\"note\":\"Same RKNN context and model, CPU-letterboxed versus RGA-letterboxed uint8 input; output differences are not labeled accuracy.\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RKNN input comparison error: " << error.what() << '\n';
        return 2;
    }
}

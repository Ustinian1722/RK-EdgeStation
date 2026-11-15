#include "multistream.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace edge {
namespace {

struct JsonValue {
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string input) : input_(std::move(input)) {}

    JsonValue parse() {
        auto result = value(0);
        whitespace();
        if (position_ != input_.size()) fail("trailing data");
        return result;
    }

private:
    [[noreturn]] void fail(const char* message) const {
        throw std::runtime_error("camera config JSON at byte " + std::to_string(position_) +
                                 ": " + message);
    }

    void whitespace() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\n' ||
                input_[position_] == '\r' || input_[position_] == '\t')) ++position_;
    }

    bool consume(char expected) {
        whitespace();
        if (position_ >= input_.size() || input_[position_] != expected) return false;
        ++position_;
        return true;
    }

    static void append_utf8(std::string& out, unsigned int codepoint) {
        if (codepoint <= 0x7f) out.push_back(static_cast<char>(codepoint));
        else if (codepoint <= 0x7ff) {
            out.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            out.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    unsigned int hex4() {
        if (position_ + 4 > input_.size()) fail("incomplete unicode escape");
        unsigned int codepoint = 0;
        for (int index = 0; index < 4; ++index) {
            const char ch = input_[position_++];
            codepoint <<= 4;
            if (ch >= '0' && ch <= '9') codepoint |= static_cast<unsigned int>(ch - '0');
            else if (ch >= 'a' && ch <= 'f') codepoint |= static_cast<unsigned int>(ch - 'a' + 10);
            else if (ch >= 'A' && ch <= 'F') codepoint |= static_cast<unsigned int>(ch - 'A' + 10);
            else fail("invalid unicode escape");
        }
        return codepoint;
    }

    std::string string() {
        if (!consume('"')) fail("expected string");
        std::string out;
        while (position_ < input_.size()) {
            const unsigned char ch = static_cast<unsigned char>(input_[position_++]);
            if (ch == '"') return out;
            if (ch < 0x20) fail("control byte in string");
            if (ch != '\\') { out.push_back(static_cast<char>(ch)); continue; }
            if (position_ >= input_.size()) fail("incomplete escape");
            switch (input_[position_++]) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    const unsigned int codepoint = hex4();
                    if (codepoint >= 0xd800 && codepoint <= 0xdfff)
                        fail("surrogate unicode escapes are not supported");
                    append_utf8(out, codepoint);
                    break;
                }
                default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue value(unsigned int depth) {
        if (depth > 64) fail("nesting limit exceeded");
        whitespace();
        if (position_ >= input_.size()) fail("expected value");
        if (input_[position_] == '"') return JsonValue{string()};
        if (input_[position_] == '{') {
            ++position_;
            JsonValue::Object object;
            whitespace();
            if (consume('}')) return JsonValue{std::move(object)};
            do {
                whitespace();
                if (position_ >= input_.size() || input_[position_] != '"') fail("expected object key");
                auto key = string();
                if (!consume(':')) fail("expected colon");
                if (!object.emplace(std::move(key), value(depth + 1)).second) fail("duplicate object key");
            } while (consume(','));
            if (!consume('}')) fail("expected closing brace");
            return JsonValue{std::move(object)};
        }
        if (input_[position_] == '[') {
            ++position_;
            JsonValue::Array array;
            whitespace();
            if (consume(']')) return JsonValue{std::move(array)};
            do { array.push_back(value(depth + 1)); } while (consume(','));
            if (!consume(']')) fail("expected closing bracket");
            return JsonValue{std::move(array)};
        }
        if (input_.compare(position_, 4, "true") == 0) {
            position_ += 4; return JsonValue{true};
        }
        if (input_.compare(position_, 5, "false") == 0) {
            position_ += 5; return JsonValue{false};
        }
        if (input_.compare(position_, 4, "null") == 0) {
            position_ += 4; return JsonValue{nullptr};
        }
        const auto start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size()) fail("invalid number");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                fail("leading zero in number");
        } else if (input_[position_] >= '1' && input_[position_] <= '9') {
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9')
                ++position_;
        } else {
            fail("invalid number");
        }
        if (position_ < input_.size() && input_[position_] == '.') {
            ++position_;
            const auto fraction_start = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (fraction_start == position_) fail("fraction requires digits");
        }
        if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
            ++position_;
            if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            const auto exponent_start = position_;
            while (position_ < input_.size() && input_[position_] >= '0' && input_[position_] <= '9') ++position_;
            if (exponent_start == position_) fail("exponent requires digits");
        }
        if (start == position_) fail("invalid value");
        try {
            const double number = std::stod(input_.substr(start, position_ - start));
            if (!std::isfinite(number)) fail("non-finite number");
            return JsonValue{number};
        } catch (const std::exception&) { fail("invalid number"); }
    }

    std::string input_;
    std::size_t position_ = 0;
};

const JsonValue::Object& object(const JsonValue& value, const char* where) {
    const auto* result = std::get_if<JsonValue::Object>(&value.value);
    if (!result) throw std::runtime_error(std::string(where) + " must be a JSON object");
    return *result;
}

const JsonValue::Array& array(const JsonValue& value, const char* where) {
    const auto* result = std::get_if<JsonValue::Array>(&value.value);
    if (!result) throw std::runtime_error(std::string(where) + " must be a JSON array");
    return *result;
}

const JsonValue* member(const JsonValue::Object& object_value, const char* key) {
    const auto found = object_value.find(key);
    return found == object_value.end() ? nullptr : &found->second;
}

std::string string_value(const JsonValue& value, const char* where) {
    const auto* result = std::get_if<std::string>(&value.value);
    if (!result) throw std::runtime_error(std::string(where) + " must be a string");
    return *result;
}

double number_value(const JsonValue& value, const char* where) {
    const auto* result = std::get_if<double>(&value.value);
    if (!result) throw std::runtime_error(std::string(where) + " must be a number");
    return *result;
}

bool bool_value(const JsonValue& value, const char* where) {
    const auto* result = std::get_if<bool>(&value.value);
    if (!result) throw std::runtime_error(std::string(where) + " must be a boolean");
    return *result;
}

bool safe_id(const std::string& id) {
    if (id.empty() || id.size() > 64 || id == "." || id == "..") return false;
    for (const unsigned char ch : id) {
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
              (ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.')) return false;
    }
    return true;
}

void optional_number(const JsonValue::Object& values, const char* key, double& destination) {
    if (const auto* value = member(values, key)) destination = number_value(*value, key);
}

}  // namespace

MultiStreamConfig load_multistream_config(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open camera config: " + path);
    std::ostringstream contents;
    contents << input.rdbuf();
    const std::string text = contents.str();
    if (text.empty() || text.size() > 1024 * 1024)
        throw std::runtime_error("camera config must contain 1..1048576 bytes");
    const auto root_value = JsonParser(text).parse();
    const auto& root = object(root_value, "config");
    const auto* cameras_value = member(root, "cameras");
    if (!cameras_value) throw std::runtime_error("camera config requires a cameras array");
    const auto& cameras = array(*cameras_value, "cameras");
    if (cameras.empty() || cameras.size() > 32)
        throw std::runtime_error("camera config must contain 1..32 camera entries");

    MultiStreamConfig result;
    std::map<std::string, bool> ids;
    bool any_enabled = false;
    for (std::size_t index = 0; index < cameras.size(); ++index) {
        const auto& fields = object(cameras[index], "camera entry");
        const auto* id_value = member(fields, "id");
        const auto* source_value = member(fields, "source");
        if (!id_value || !source_value) throw std::runtime_error("each camera requires id and source");
        MultiCameraConfig camera;
        camera.id = string_value(*id_value, "camera id");
        camera.source = string_value(*source_value, "camera source");
        if (!safe_id(camera.id)) throw std::runtime_error("camera id must use 1..64 safe ASCII characters");
        if (!ids.emplace(camera.id, true).second) throw std::runtime_error("camera ids must be unique");
        if (camera.source.empty() || camera.source.size() > 2048)
            throw std::runtime_error("camera source must contain 1..2048 characters");
        if (const auto* value = member(fields, "enabled")) camera.enabled = bool_value(*value, "enabled");
        optional_number(fields, "max_fps", camera.max_fps);
        if (camera.max_fps < 0 || camera.max_fps > 120)
            throw std::runtime_error("max_fps must be between 0 and 120");
        if (const auto* value = member(fields, "weight")) {
            const double parsed = number_value(*value, "weight");
            if (parsed < 1 || parsed > 64 || std::floor(parsed) != parsed)
                throw std::runtime_error("weight must be an integer between 1 and 64");
            camera.weight = static_cast<unsigned int>(parsed);
        }
        if (const auto* value = member(fields, "queue_capacity")) {
            const double parsed = number_value(*value, "queue_capacity");
            if (parsed < 1 || parsed > 8 || std::floor(parsed) != parsed)
                throw std::runtime_error("queue_capacity must be an integer between 1 and 8");
            camera.queue_capacity = static_cast<std::size_t>(parsed);
        }
        if (const auto* value = member(fields, "tasks")) {
            camera.tasks.clear();
            std::map<std::string, bool> task_names;
            for (const auto& task : array(*value, "tasks")) {
                const auto name = string_value(task, "task name");
                if (name != "ppe" && name != "roi")
                    throw std::runtime_error("supported tasks are ppe and roi; fire/smoke is not ready");
                if (!task_names.emplace(name, true).second) throw std::runtime_error("duplicate camera task");
                camera.tasks.push_back(name);
            }
            if (camera.tasks.empty()) throw std::runtime_error("each enabled camera needs at least one task");
        }
        if (const auto* value = member(fields, "roi")) {
            const auto& roi = array(*value, "roi");
            if (roi.size() != 4) throw std::runtime_error("roi must be [x,y,width,height]");
            camera.roi_x = number_value(roi[0], "roi x");
            camera.roi_y = number_value(roi[1], "roi y");
            camera.roi_width = number_value(roi[2], "roi width");
            camera.roi_height = number_value(roi[3], "roi height");
            if (camera.roi_x < 0 || camera.roi_y < 0 || camera.roi_width <= 0 ||
                camera.roi_height <= 0 || camera.roi_x + camera.roi_width > 1 ||
                camera.roi_y + camera.roi_height > 1)
                throw std::runtime_error("camera roi must be a normalized rectangle inside the frame");
        }
        any_enabled = any_enabled || camera.enabled;
        result.cameras.push_back(std::move(camera));
    }
    if (!any_enabled) throw std::runtime_error("at least one camera must be enabled");
    return result;
}

WeightedFairScheduler::WeightedFairScheduler(std::vector<unsigned int> weights)
    : weights_(std::move(weights)), current_(weights_.size(), 0) {
    if (weights_.empty()) throw std::invalid_argument("scheduler needs at least one camera");
    for (const auto weight : weights_) {
        if (weight == 0 || weight > 64) throw std::invalid_argument("scheduler weight must be 1..64");
    }
}

int WeightedFairScheduler::select(const std::vector<bool>& ready) {
    if (ready.size() != weights_.size()) throw std::invalid_argument("ready vector size mismatch");
    long long total = 0;
    int selected = -1;
    for (std::size_t index = 0; index < ready.size(); ++index) {
        if (!ready[index]) continue;
        total += weights_[index];
        current_[index] += weights_[index];
        if (selected < 0 || current_[index] > current_[static_cast<std::size_t>(selected)])
            selected = static_cast<int>(index);
    }
    if (selected >= 0) current_[static_cast<std::size_t>(selected)] -= total;
    return selected;
}

void WeightedFairScheduler::reset() {
    std::fill(current_.begin(), current_.end(), 0);
}

std::string redact_source(const std::string& source) {
    const auto scheme = source.find("://");
    if (scheme == std::string::npos) return source;
    const auto authority_start = scheme + 3;
    const auto authority_end = source.find_first_of("/?#", authority_start);
    const auto end = authority_end == std::string::npos ? source.size() : authority_end;
    const auto at = source.rfind('@', end);
    if (at != std::string::npos && at >= authority_start)
        return source.substr(0, authority_start) + source.substr(at + 1, end - at - 1);
    return source.substr(0, end);
}

}  // namespace edge

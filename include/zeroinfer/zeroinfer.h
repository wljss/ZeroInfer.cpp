#pragma once

#include <cstdint>
#include <string>

namespace zeroinfer
{
struct InferenceOptions
{
    std::string model_path = "stories15M.bin";
    std::string tokenizer_path = "tokenizer.bin";
    std::string prompt = "Long long ago";
    int steps = 0;
    float temperature = 0.95f;
    float top_p = 0.9f;
    int threads = 0;
    std::uint64_t seed = 233333;
};

void run(const InferenceOptions& options);
}

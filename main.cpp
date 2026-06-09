#include "zeroinfer/zeroinfer.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{
void print_usage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Options:\n"
        << "  --model PATH          Model checkpoint (default: stories15M.bin)\n"
        << "  --tokenizer PATH      Tokenizer file (default: tokenizer.bin)\n"
        << "  --prompt TEXT         Prompt text (default: \"Long long ago\")\n"
        << "  --steps N             Total forward steps; 0 uses model maximum (default: 0)\n"
        << "  --temperature VALUE   Sampling temperature; 0 uses argmax (default: 0.95)\n"
        << "  --top-p VALUE         Nucleus sampling threshold in [0, 1] (default: 0.9)\n"
        << "  --threads N           OpenMP thread count; 0 uses runtime default (default: 0)\n"
        << "  --seed N              Random seed (default: 233333)\n"
        << "  -h, --help            Show this help message\n";
}

std::string require_value(int& index,int argc,char* argv[],const std::string& option)
{
    if(index+1>=argc)
    {
        throw std::invalid_argument("missing value for "+option);
    }
    return argv[++index];
}

int parse_int(const std::string& value,const std::string& option)
{
    std::size_t parsed=0;
    long long result=0;
    try
    {
        result=std::stoll(value,&parsed);
    }
    catch(const std::exception&)
    {
        throw std::invalid_argument("invalid integer for "+option+": "+value);
    }

    if(parsed!=value.size()||result<std::numeric_limits<int>::min()||result>std::numeric_limits<int>::max())
    {
        throw std::invalid_argument("invalid integer for "+option+": "+value);
    }
    return static_cast<int>(result);
}

std::uint64_t parse_seed(const std::string& value)
{
    if(!value.empty()&&value.front()=='-')
    {
        throw std::invalid_argument("invalid integer for --seed: "+value);
    }

    std::size_t parsed=0;
    unsigned long long result=0;
    try
    {
        result=std::stoull(value,&parsed);
    }
    catch(const std::exception&)
    {
        throw std::invalid_argument("invalid integer for --seed: "+value);
    }

    if(parsed!=value.size())
    {
        throw std::invalid_argument("invalid integer for --seed: "+value);
    }
    return static_cast<std::uint64_t>(result);
}

float parse_float(const std::string& value,const std::string& option)
{
    std::size_t parsed=0;
    float result=0.0f;
    try
    {
        result=std::stof(value,&parsed);
    }
    catch(const std::exception&)
    {
        throw std::invalid_argument("invalid number for "+option+": "+value);
    }

    if(parsed!=value.size())
    {
        throw std::invalid_argument("invalid number for "+option+": "+value);
    }
    return result;
}
}

int main(int argc,char* argv[])
{
    try
    {
        zeroinfer::InferenceOptions options;

        for(int i=1;i<argc;++i)
        {
            const std::string argument=argv[i];

            if(argument=="-h"||argument=="--help")
            {
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if(argument=="--model")
            {
                options.model_path=require_value(i,argc,argv,argument);
            }
            else if(argument=="--tokenizer")
            {
                options.tokenizer_path=require_value(i,argc,argv,argument);
            }
            else if(argument=="--prompt")
            {
                options.prompt=require_value(i,argc,argv,argument);
            }
            else if(argument=="--steps")
            {
                options.steps=parse_int(require_value(i,argc,argv,argument),argument);
            }
            else if(argument=="--temperature")
            {
                options.temperature=parse_float(require_value(i,argc,argv,argument),argument);
            }
            else if(argument=="--top-p")
            {
                options.top_p=parse_float(require_value(i,argc,argv,argument),argument);
            }
            else if(argument=="--threads")
            {
                options.threads=parse_int(require_value(i,argc,argv,argument),argument);
            }
            else if(argument=="--seed")
            {
                options.seed=parse_seed(require_value(i,argc,argv,argument));
            }
            else
            {
                throw std::invalid_argument("unknown option: "+argument);
            }
        }

        zeroinfer::run(options);
        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr<<"error: "<<error.what()<<"\n";
        std::cerr<<"Try --help for usage.\n";
        return EXIT_FAILURE;
    }
}

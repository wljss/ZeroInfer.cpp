#include "internal/ops.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
int parse_positive_int(const char* value,const std::string& name)
{
    std::size_t parsed=0;
    int result=0;

    try
    {
        result=std::stoi(value,&parsed);
    }
    catch(const std::exception&)
    {
        throw std::invalid_argument("invalid "+name+": "+value);
    }

    if(parsed!=std::string(value).size()||result<=0)
    {
        throw std::invalid_argument("invalid "+name+": "+value);
    }

    return result;
}

std::vector<float> read_floats(const std::string& path,std::size_t count)
{
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)
    {
        throw std::runtime_error("could not open input file: "+path);
    }

    const auto expected_size=static_cast<std::streamoff>(count*sizeof(float));
    if(input.tellg()!=expected_size)
    {
        throw std::runtime_error("unexpected input size: "+path);
    }

    std::vector<float> values(count);
    input.seekg(0);
    if(!input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected_size)
    ))
    {
        throw std::runtime_error("could not read input file: "+path);
    }

    return values;
}

void write_floats(const std::string& path,const std::vector<float>& values)
{
    std::ofstream output(path,std::ios::binary);
    if(!output)
    {
        throw std::runtime_error("could not open output file: "+path);
    }

    const auto size=static_cast<std::streamsize>(values.size()*sizeof(float));
    if(!output.write(reinterpret_cast<const char*>(values.data()),size))
    {
        throw std::runtime_error("could not write output file: "+path);
    }
}

void run_matmul(int argc,char* argv[])
{
    if(argc!=7)
    {
        throw std::invalid_argument(
            "matmul expects: x.bin weights.bin out.bin in_features out_features"
        );
    }

    const int in_features=parse_positive_int(argv[5],"in_features");
    const int out_features=parse_positive_int(argv[6],"out_features");
    const auto x=read_floats(argv[2],static_cast<std::size_t>(in_features));
    const auto weights=read_floats(
        argv[3],
        static_cast<std::size_t>(in_features)*out_features
    );
    std::vector<float> out(static_cast<std::size_t>(out_features));

    zeroinfer::internal::matmul(
        out.data(),
        x.data(),
        weights.data(),
        in_features,
        out_features
    );
    write_floats(argv[4],out);
}

void run_rmsnorm(int argc,char* argv[])
{
    if(argc!=6)
    {
        throw std::invalid_argument(
            "rmsnorm expects: x.bin weight.bin out.bin size"
        );
    }

    const int size=parse_positive_int(argv[5],"size");
    const auto x=read_floats(argv[2],static_cast<std::size_t>(size));
    const auto weight=read_floats(argv[3],static_cast<std::size_t>(size));
    std::vector<float> out(static_cast<std::size_t>(size));

    zeroinfer::internal::rmsnorm(
        out.data(),
        x.data(),
        weight.data(),
        size
    );
    write_floats(argv[4],out);
}

void run_softmax(int argc,char* argv[])
{
    if(argc!=5)
    {
        throw std::invalid_argument("softmax expects: values.bin out.bin size");
    }

    const int size=parse_positive_int(argv[4],"size");
    auto values=read_floats(argv[2],static_cast<std::size_t>(size));

    zeroinfer::internal::softmax(values.data(),size);
    write_floats(argv[3],values);
}
}

int main(int argc,char* argv[])
{
    try
    {
        if(argc<2)
        {
            throw std::invalid_argument("missing operation");
        }

        const std::string operation=argv[1];
        if(operation=="matmul")
        {
            run_matmul(argc,argv);
        }
        else if(operation=="rmsnorm")
        {
            run_rmsnorm(argc,argv);
        }
        else if(operation=="softmax")
        {
            run_softmax(argc,argv);
        }
        else
        {
            throw std::invalid_argument("unknown operation: "+operation);
        }

        return EXIT_SUCCESS;
    }
    catch(const std::exception& error)
    {
        std::cerr<<"error: "<<error.what()<<"\n";
        return EXIT_FAILURE;
    }
}

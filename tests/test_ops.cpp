#include "internal/ops.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures=0;

void expect_near(
    float actual,
    float expected,
    float tolerance,
    const std::string& description
)
{
    if(!std::isfinite(actual)||std::fabs(actual-expected)>tolerance)
    {
        std::cerr
            <<"FAIL: "<<description
            <<" (expected "<<expected<<", got "<<actual<<")\n";
        ++failures;
    }
}

void expect_vector_near(
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float tolerance,
    const std::string& description
)
{
    if(actual.size()!=expected.size())
    {
        std::cerr
            <<"FAIL: "<<description
            <<" (expected size "<<expected.size()
            <<", got "<<actual.size()<<")\n";
        ++failures;
        return;
    }

    for(std::size_t i=0;i<actual.size();++i)
    {
        expect_near(
            actual[i],
            expected[i],
            tolerance,
            description+"["+std::to_string(i)+"]"
        );
    }
}

void test_matmul_non_square()
{
    const std::vector<float> x={0.5f,-1.0f,2.0f,0.25f};
    const std::vector<float> weights={
        1.0f,2.0f,3.0f,4.0f,
        -2.0f,0.0f,1.0f,3.0f,
        0.25f,0.5f,-0.5f,2.0f
    };
    std::vector<float> out(3,0.0f);

    zeroinfer::internal::matmul(
        out.data(),
        x.data(),
        weights.data(),
        4,
        3
    );

    expect_vector_near(
        out,
        {5.5f,1.75f,-0.875f},
        1e-6f,
        "matmul non-square"
    );
}

void test_rmsnorm_reference()
{
    const std::vector<float> x={1.0f,-2.0f,3.0f,-4.0f};
    const std::vector<float> weight={1.0f,0.5f,-1.0f,2.0f};
    std::vector<float> out(x.size(),0.0f);

    zeroinfer::internal::rmsnorm(
        out.data(),
        x.data(),
        weight.data(),
        static_cast<int>(x.size())
    );

    const float scale=1.0f/std::sqrt(7.5f+1e-5f);
    expect_vector_near(
        out,
        {scale,-scale,-3.0f*scale,-8.0f*scale},
        1e-6f,
        "rmsnorm reference"
    );
}

void test_rmsnorm_in_place()
{
    std::vector<float> values={1.0f,-2.0f,3.0f,-4.0f};
    const std::vector<float> weight={1.0f,0.5f,-1.0f,2.0f};
    const float scale=1.0f/std::sqrt(7.5f+1e-5f);

    zeroinfer::internal::rmsnorm(
        values.data(),
        values.data(),
        weight.data(),
        static_cast<int>(values.size())
    );

    expect_vector_near(
        values,
        {scale,-scale,-3.0f*scale,-8.0f*scale},
        1e-6f,
        "rmsnorm in-place"
    );
}

void test_softmax_reference()
{
    std::vector<float> values={1.0f,2.0f,3.0f};

    zeroinfer::internal::softmax(
        values.data(),
        static_cast<int>(values.size())
    );

    expect_vector_near(
        values,
        {0.09003057f,0.24472848f,0.66524094f},
        1e-6f,
        "softmax reference"
    );

    const float probability_sum=values[0]+values[1]+values[2];
    expect_near(probability_sum,1.0f,1e-6f,"softmax probability sum");
}

void test_softmax_large_logits()
{
    std::vector<float> values={1000.0f,1001.0f,1002.0f};

    zeroinfer::internal::softmax(
        values.data(),
        static_cast<int>(values.size())
    );

    expect_vector_near(
        values,
        {0.09003057f,0.24472848f,0.66524094f},
        1e-6f,
        "softmax large logits"
    );
}
}

int main(int argc,char* argv[])
{
    const std::string selected=argc>1?argv[1]:"all";

    if(selected=="all"||selected=="matmul")
    {
        test_matmul_non_square();
    }
    if(selected=="all"||selected=="rmsnorm")
    {
        test_rmsnorm_reference();
        test_rmsnorm_in_place();
    }
    if(selected=="all"||selected=="softmax")
    {
        test_softmax_reference();
        test_softmax_large_logits();
    }
    if(
        selected!="all"&&
        selected!="matmul"&&
        selected!="rmsnorm"&&
        selected!="softmax"
    )
    {
        std::cerr<<"Unknown test group: "<<selected<<"\n";
        return 1;
    }

    if(failures!=0)
    {
        std::cerr<<failures<<" assertion(s) failed\n";
        return 1;
    }

    std::cout<<"All operator tests passed\n";
    return 0;
}

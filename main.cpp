#include<iostream>
#include<vector>
#include<sys/stat.h>
#include<unistd.h>
#include<sys/mman.h>
#include<fcntl.h>
#include<array>
#include<fstream>
#include<cstdint>
#include<string>
#include<algorithm>

struct Config //Llama2.c导出的二进制文件头部的28个字节是配置参数
{
    int dim; //Transformer dimension
    int hidden_dim; //FFN层隐藏维度
    int n_layers; //层数
    int n_q_heads; //Query 头数
    int n_kv_heads; //Key/Value 头数,GQA和上面头数不一样
    int vocab_size; //词表大小
    int seq_len; //最大序列长度
};
struct TransformerWeights //网络权重
{
    float* token_embedding_table; //维度为(vocab_size,dim)，词嵌入表
    float* rms_att_weight; //维度为(n_layers, dim)，attention子层前面的RMSNorm权重
    float* rms_ffn_weight; //维度为(n_layers, dim)，FFN子层前面的RMSNorm权重
    float* Wq; //维度为(n_layers, dim, dim( = n_q_heads * head_size))
    float* Wk; //维度为(n_layers, dim, n_kv_heads * head_size)
    float* Wv; //维度为(n_layers, dim, n_kv_heads * head_size)
    float* Wo; //维度为(n_layers, dim, dim( = n_q_heads * head_size))
    float* W1; //维度为(n_layers, hidden_dim , dim),SwiGLU上投影矩阵,把dim维投影到hidden_dim维
    float* W2; //维度为(n_layers, dim , hidden_dim),SwiGLU下投影矩阵,把FFN中间维度投影回Transformer主干维度
    float* W3; //维度为(n_layers, hidden_dim , dim),SwiGLU上投影矩阵,gate分支
    float* rms_final_weight; //所有Transformer层结束后的最终RMSNorm权重
    float* wcls; //最终分类器权重，把最终hidden state转成词表大小的logits
};

struct RunState
{
    // vector相比new[]可以自己释放，性能上也差不多
    std::vector<float> x; //各个变量的解释见allocate
    std::vector<float> xb; 
    std::vector<float> xb2; 
    std::vector<float> hb; 
    std::vector<float> hb2; 
    std::vector<float> q; 
    float* k = nullptr; //后面会指向cache
    float* v = nullptr; //后面会指向cache
    std::vector<float> key_cache; 
    std::vector<float> value_cache; 
    std::vector<float> att; 
    std::vector<float> logits; 
    

    RunState()=default; //默认构造函数
    explicit RunState(const Config &p){allocate(p);} //带参数的构造函数。explicit的作用是禁止隐式类型转换
    
    void allocate(const Config &p)
    {
        const int kv_dim = p.dim / p.n_q_heads * p.n_kv_heads ;

        x.assign(static_cast<size_t>(p.dim), 0.0f); //TODO:这些维度怎么来的？
        xb.assign(static_cast<size_t>(p.dim), 0.0f);
        xb2.assign(static_cast<size_t>(p.dim), 0.0f);

        hb.assign(static_cast<size_t>(p.hidden_dim), 0.0f);
        hb2.assign(static_cast<size_t>(p.hidden_dim), 0.0f);

        q.assign(static_cast<size_t>(p.dim), 0.0f); //query,dim=n_q_heads*head_size

        key_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f); //每层每个位置缓存kv_dim维key
        value_cache.assign(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim, 0.0f); //每层每个位置缓存kv_dim维value

        att.assign(static_cast<size_t>(p.n_q_heads) * p.seq_len, 0.0f); //每个query head对历史seq_len个位置的attention分数

        logits.assign(p.vocab_size, 0.0f); //输出词表上每个token的未归一化分数
    }
};

struct Transformer
{
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd; //checkpoint文件的文件描述符
    float* data; //指向checkpoint文件映射到内存的起点
    ssize_t file_size; //checkpoint文件字节大小

    ~Transformer() 
    {
        if (data != nullptr && data != MAP_FAILED) 
        {
            munmap(data, file_size);
        }
        if (fd != -1) 
        {
            close(fd);
        }
    }
};

void assign_weights(TransformerWeights &weights,float* weights_ptr,const Config &config,const bool shared_weights)
{
    const size_t n_layers = static_cast<size_t>(config.n_layers); //这样做是为了防止int会溢出
    //赋值权重
    weights.token_embedding_table = weights_ptr;
    weights_ptr += config.vocab_size * config.dim;

    weights.rms_att_weight = weights_ptr;
    weights_ptr += n_layers * config.dim;

    int head_size=config.dim / config.n_q_heads; //head_size为每个头的维度
    weights.Wq = weights_ptr;
    weights_ptr += n_layers * config.dim * config.dim;

    weights.Wk = weights_ptr;
    weights_ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.Wv = weights_ptr;
    weights_ptr += n_layers * config.dim * (config.n_kv_heads * head_size);

    weights.Wo = weights_ptr;
    weights_ptr += n_layers * config.dim * config.dim;

    weights.rms_ffn_weight = weights_ptr;
    weights_ptr += n_layers * config.dim;

    weights.W1 = weights_ptr;
    weights_ptr += n_layers * config.hidden_dim * config.dim;

    weights.W2 = weights_ptr;
    weights_ptr += n_layers * config.dim * config.hidden_dim;

    weights.W3 = weights_ptr;
    weights_ptr += n_layers * config.hidden_dim * config.dim;

    weights.rms_final_weight = weights_ptr;
    weights_ptr += config.dim;
    const size_t rope_table_size = static_cast<size_t>(config.seq_len) * head_size / 2;

    // 跳过的是旧的ROPE用到的部分，这里用不到了
    weights_ptr += rope_table_size;
    weights_ptr += rope_table_size;

    weights.wcls = weights.token_embedding_table;// TODO:源码好像用的是share部分

}

struct TokenIndex //用来根据字符串找到token id
{
    std::string str;
    int id;
    friend bool operator <(const TokenIndex& a, const TokenIndex& b) // 按照字典序排序
    { 
        return a.str < b.str;
    }
};

struct Tokenizer 
{
    std::vector<std::string> vocab; //词表数组，根据token id找到字符串
    std::vector<float> vocab_scores; //用于BPE合并
    std::vector<TokenIndex> sorted_vocab; //按照字符串排序，encode阶段能快速根据字符串找到token id

    int vocab_size = 0; //词表大小
    std::uint32_t max_token_length = 0; //词表中最长token字符串长度TODO:

    std::array<unsigned char, 512> byte_pieces{}; //储存256个单字节字符串，每个字符串占2个字节，例如'A'后面接一个'\0'
};

void build_tokenizer(Tokenizer& tokenizer, const std::string& tokenizer_path, int vocab_size) 
{
    //初始化
    tokenizer.vocab.resize(vocab_size);
    tokenizer.vocab_scores.resize(vocab_size);
    tokenizer.sorted_vocab.clear();
    tokenizer.vocab_size = vocab_size;

    for (int i = 0; i < 256; i++) {
        tokenizer.byte_pieces[i * 2] = static_cast<unsigned char>(i);
        tokenizer.byte_pieces[i * 2 + 1] = '\0';
    }

    //读文件
    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) 
    {
        throw std::runtime_error("couldn't load " + tokenizer_path);
    }

    auto read_exact = [&](char* dst, std::size_t size) //lambda表达式,这个&把外部变量按引用捕获进来了
    {
        if (!file.read(dst, static_cast<std::streamsize>(size))) //static_cast是显示普通类型转换
        {
            throw std::runtime_error("failed read");
        }
    };

    std::int32_t max_len_from_file = 0;
    read_exact(reinterpret_cast<char*>(&max_len_from_file), sizeof(std::int32_t)); //reinterpret_cast是底层内存重新解释的转换
    tokenizer.max_token_length = static_cast<std::uint32_t>(max_len_from_file);

    for (int i = 0; i < vocab_size; i++) 
    {
        read_exact(reinterpret_cast<char*>(&tokenizer.vocab_scores[i]), sizeof(float));

        std::int32_t len = 0;
        read_exact(reinterpret_cast<char*>(&len), sizeof(std::int32_t));

        if (len < 0) 
        {
            throw std::runtime_error("invalid token length");
        }

        std::string token(len, '\0');
        read_exact(&token[0], static_cast<std::size_t>(len));

        tokenizer.vocab[i] = std::move(token); //通过move移动赋值，不写的话就是拷贝赋值
    }
}

struct ProbIndex // top-p采样时用
{
    float prob; //token的概率
    int index; //token的id
}; 

struct Sampler //采样器，把模型输出的logits转换成下一个token
{
    int vocab_size; //词表大小
    std::vector<ProbIndex> probindex; //top-p采样时用的临时数组
    float temperature; //改变logits分布的尖锐程度，logits除以temperature，temperature越小越尖锐保守，temperature越大越平坦随机
    float topp; //核采样参数
    unsigned long long rng_state; //随机数生成器的状态

    Sampler()=default; //默认构造函数
    explicit Sampler(int vocab_size_,float temperature_,float topp_,unsigned long long rng_seed_) : vocab_size(vocab_size_),probindex(vocab_size_),temperature(temperature_),topp(topp_),rng_state(rng_seed_)//带参数的构造函数。explicit的作用是禁止隐式类型转换
    {
    } 
};

void matmul(float* xout,const float* x,const float* w,int n,int d) // W(d,n)矩阵乘法x(n,)得到的结果存到xout (d,)
{
    #pragma omp parallel for //并行计算i
    for(int i=0;i<d;++i) 
    {
        const float* row=w+i*n; //行号

        float val=0.0f;
        for (int j=0;j<n;++j) //j是串行计算的 
        {
            val+=row[j]*x[j];
        }

        xout[i]=val;
    }
}
void ensure_sorted_vocab(Tokenizer& tokenizer)
{
    if (!tokenizer.sorted_vocab.empty()) 
    {
        return;
    }

    tokenizer.sorted_vocab.reserve(static_cast<size_t>(tokenizer.vocab_size));

    for (int i=0;i<tokenizer.vocab_size;++i) 
    {
        tokenizer.sorted_vocab.push_back(TokenIndex{tokenizer.vocab[i],i});
    }

    std::sort(tokenizer.sorted_vocab.begin(),tokenizer.sorted_vocab.end());
}
int str_lookup(const std::string& str,const Tokenizer& tokenizer) //二分查找str对应得token id
{
    auto it = std::lower_bound(tokenizer.sorted_vocab.begin(),tokenizer.sorted_vocab.end(),(TokenIndex){str,-1}); //TokenIndex默认比较字符串，赋值个-1没问题

    if(it!=tokenizer.sorted_vocab.end()&&it->str==str) 
    {
        return it->id;
    }

    return -1;
} 
std::vector<int> encode(Tokenizer& tokenizer,const std::string& text,bool bos,bool eos) //这里的bos和eos是是否加入开始/结束标记
{
    ensure_sorted_vocab(tokenizer);

    std::vector<int> tokens;
    tokens.reserve(text.size()+3); //+3是BOS和EOS和dummy_prefix

    if (bos) //可选添加BOS token，BOS=1
    {
        tokens.push_back(1);
    }

    if (!text.empty()) //SentencePiece风格的dummy prefix,非空文本前加一个空格 token 
    {
        int dummy_prefix=str_lookup(" ",tokenizer);
        if(dummy_prefix==-1) 
        {
            throw std::runtime_error("dummy prefix token ' ' not found in vocabulary");
        }

        tokens.push_back(dummy_prefix);
    }

    //按UTF-8 codepoint初步编码。这是因为中文字符和表情等会占用多个字节，规律如下
    /*
        1字节字符: 0xxxxxxx
        2字节字符: 110xxxxx 10xxxxxx
        3字节字符: 1110xxxx 10xxxxxx 10xxxxxx
        4字节字符: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    */
    std::string str_buffer; //用来处理UTF-8
    str_buffer.reserve(4); //UTF-8单个codepoint最多4字节

    for(size_t pos=0;pos<text.size();++pos) 
    {
        unsigned char byte=text[pos];
        
        if((byte&0xC0)!=0x80) //如果当前字节不是UTF-8 continuation byte，则开始新的codepoint
        {
            str_buffer.clear();
        }

        str_buffer.push_back(static_cast<char>(byte));

        bool next_is_continuation = false;
        if(pos+1<text.size()) 
        {
            next_is_continuation=((text[pos+1]&0xC0)==0x80);
        }

        
        if(next_is_continuation && str_buffer.size()<4) //如果下一个字节还是continuation byte，并且当前codepoint长度还没超过4，说明这个UTF-8 codepoint还没读完，继续累积
        {
            continue;
        }
        else //否则说明已经拿到一个完整codepoint，尝试在词表中查找
        {
            int id=str_lookup(str_buffer,tokenizer);

            if(id!=-1) //找到了
            {
                tokens.push_back(id);
            } 
            else //没找到，退化成按字节编码
            {
                
                for (unsigned char b : str_buffer) 
                {
                    tokens.push_back(static_cast<int>(b) + 3); //每个原始字节映射到byte+3是因为前三个一般是<unk> BOS EOS
                }
            }

            str_buffer.clear();
        }
    }

    
    while (true) //BPE合并,每轮选择vocab_scores最高的可合并pair
    {
        float best_score=-1e10f;
        int best_id=-1; //token id
        int best_pos=-1; //在token序列中的位置

        for(int i=0;i<static_cast<int>(tokens.size())-1;++i) 
        {
            const std::string& left=tokenizer.vocab[tokens[i]];
            const std::string& right=tokenizer.vocab[tokens[i+1]];

            std::string merged=left+right;

            int id=str_lookup(merged,tokenizer);

            if(id!=-1&&tokenizer.vocab_scores[id]>best_score) 
            {
                best_score=tokenizer.vocab_scores[id];
                best_id=id;
                best_pos=i;
            }
        }

        if(best_pos==-1) //所有相邻pair的合并string都在词表中找不到
        {
            break;
        }
        
        tokens[best_pos]=best_id; //把tokens[best_pos]和tokens[best_pos+1]合并成best_id
        tokens.erase(tokens.begin()+best_pos+1);
    }

    
    if(eos) //可选添加EOS token，EOS=2
    {
        tokens.push_back(2);
    }

    return tokens;
}



void generate(Transformer &transformer,Tokenizer &tokenizer,Sampler &sampler,const std::string &prompt,int steps)
{
    std::vector<int>prompt_tokens=encode(tokenizer,prompt,true,false); //把prompt编码成token序列

    for(auto token : prompt_tokens)
    {
        printf("token : %d\n",token);
    }
}

bool read_checkpoint_header_and_map(Config &config,const std::string filename,float* &data,int &fd,ssize_t &file_size)
{
    struct stat sb; //用于描述文件的元数据
    if(stat(filename.c_str(),&sb)==-1) //返回0表示正确填充，-1出错
    {
        throw std::runtime_error("Failed to stat file\n"); 
    }

    fd=open(filename.c_str(),O_RDONLY); //O_RDONLY只读，O_WRONLY只写，O_RDWR可读可写
    if(fd==-1)
    {
        throw std::runtime_error("Failed to open file\n"); 
    }

    //mmap零CPU拷贝,访问数据时才由DMA把磁盘页加载到页缓存
    file_size=sb.st_size;
    data=(float*)mmap(NULL,file_size,PROT_READ,MAP_PRIVATE,fd,0); //各个参数分别是：期望映射到的虚拟地址、映射长度、内存访问权限（可读/可写/可执行）、映射类型标志（私有映射，修改不会同步回磁盘原文件）、文件描述符、文件内偏移量，成功返回映射区起始虚拟地址指针，失败返回-1
    if(data==MAP_FAILED)
    {
        throw std::runtime_error("mmap failed\n"); 
    }


    int* header=(int*)data;
    config.dim = header[0];
    config.hidden_dim = header[1];
    config.n_layers = header[2];
    config.n_q_heads = header[3];
    config.n_kv_heads = header[4];
    config.vocab_size = header[5];
    config.seq_len = header[6];

    std::cout<<"dim: "<<config.dim<<"\n"<<"vocab_size: "<<config.vocab_size<<"\n";

    return 1;
}
void build_transformer(Transformer& transformer, const std::string& checkpoint_path)
{
    bool shared_weights = read_checkpoint_header_and_map(transformer.config,checkpoint_path,transformer.data,transformer.fd,transformer.file_size);

    transformer.state.allocate(transformer.config);

    float* weights_ptr = transformer.data + sizeof(Config) / sizeof(float);

    assign_weights(transformer.weights,weights_ptr,transformer.config,shared_weights); //赋值模型权重
}
int main()
{
    Transformer transformer;
    build_transformer(transformer,"stories15M.bin");

    Tokenizer tokenizer;
    build_tokenizer(tokenizer,"tokenizer.bin",transformer.config.vocab_size);
    
    Sampler sampler(transformer.config.vocab_size,0.95,0.9,233333); //构造采样器

    generate(transformer,tokenizer,sampler,"Long long ago",transformer.config.seq_len);

    return 0;
}
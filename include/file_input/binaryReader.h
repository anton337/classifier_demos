#ifndef BINARY_READER
#define BINARY_READER

template<typename T>
struct binaryReader
{
    void readBinary(std::string filename, std::vector<char> & out)
    {
      std::ifstream input( filename.c_str(), std::ios::binary );
      // copies all data longo buffer
      std::vector<char> tmp((
        std::istreambuf_iterator<char>(input)), 
        (std::istreambuf_iterator<char>()));
      for(int k=0;k<tmp.size();k++)
        out.push_back(tmp[k]);
    }
};

#endif


#ifndef BINARY_READER
#define BINARY_READER

template<typename T>
struct binaryReader
{
    void readBinary(std::string filename, std::vector<char> & out)
    {
      std::cout << "reading file: " << filename << std::endl;
      std::ifstream input( filename.c_str(), std::ios::binary );
      // copies all data longo buffer
      std::vector<char> tmp((
        std::istreambuf_iterator<char>(input)), 
        (std::istreambuf_iterator<char>()));
      for(int k=0;k<tmp.size();k++)
        out.push_back(tmp[k]);
    }
    long get_size()
    {
      std::cout << "size:" << size << std::endl;
      return size;
    }
    long size;
    T * readBinary(long offset,long nx,long ny,std::string filename,std::size_t max_num = -1)
    {
      std::vector<char> out;
      readBinary(filename,out);
      size = out.size()-offset;
      if(max_num<size)
      {
        size = max_num * nx * ny;
      }
      T * ret = new T[size];
      for(long i=offset;i<size;i++)
      {
        ret[i-offset] = (T)(unsigned char)(out[i])/256.0;
      }
      T * ret_transpose = new T[size];
      for(long i=0,k=0;i<size/(nx*ny)&&i<max_num;i++)
      {
        for(long x=0;x<nx;x++)
        {
          for(long y=0;y<ny;y++,k++)
          {
            ret_transpose[k] = ret[i*nx*ny+nx-1-x+nx*y];
          }
        }
      }
      delete [] ret;
      return ret_transpose;
    }
    char * readBinaryChars(long offset,std::string filename)
    {
      std::vector<char> out;
      readBinary(filename,out);
      size = out.size()-offset;
      char * ret = new char[size];
      for(long i=offset;i<size;i++)
      {
        ret[i-offset] = out[i];
      }
      return ret;
    }
};

#endif


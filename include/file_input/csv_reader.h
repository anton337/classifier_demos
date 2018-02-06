#ifndef CSV_READER_H
#define CSV_READER_H

template<typename T>
struct csvReader
{

  long size;

  long get_size(){return size;}

  T * read_data_floating_point(std::string filename,long col_index,std::size_t num = -1,bool ignore_null = true)
  {
    std::vector<T> dat;
    std::ifstream infile(filename.c_str());
    std::string line;
    long count = 0;
    while (std::getline(infile, line) && count < num)
    {
      boost::replace_all(line,","," ");
      std::stringstream iss(line);
      std::string token;
      for(long col=0;col<=col_index;col++)
      {
        iss >> token;
      }
      boost::erase_all(token,",");
      T val = atof(token.c_str());
      if(ignore_null)
      {
        if(val < 1e-5)
        {
          continue;
        }
      }
      count++;
      dat.push_back(atof(token.c_str()));
    }
    infile.close();
    T * ret = new T[dat.size()];
    size = dat.size();
    for(long i=0;i<dat.size();i++)
    {
      ret[i] = dat[i];
    }
    dat.clear();
    return ret;
  }

};

#endif


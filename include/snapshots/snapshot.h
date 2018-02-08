#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include <fstream>
#include <iostream>

    template<typename T>
    void dump_to_file ( RBM<T> * rbm
                      , std::string filename
                      , bool quiet=false
                      )
    {
        if(!quiet)
          std::cout << "dump to file:" << filename << std::endl;
        std::ofstream myfile (filename.c_str(),std::ios::out);
        if (myfile.is_open())
        {
          myfile << "#v" << std::endl;
          myfile << rbm->v << std::endl;
          myfile << "#h" << std::endl;
          myfile << rbm->h << std::endl;
          myfile << "#b" << std::endl;
          for(long v=0;v<rbm->v;v++)
          {
            myfile << rbm->b[v] << " ";
          }
          myfile << std::endl;
          myfile << "#c" << std::endl;
          for(long h=0;h<rbm->h;h++)
          {
            myfile << rbm->c[h] << " ";
          }
          myfile << std::endl;
          myfile << "#W" << std::endl;
          for(long k=0;k<rbm->h*rbm->v;k++)
          {
            myfile << rbm->W[k] << " ";
          }
          myfile << std::endl;
          myfile.close();
        }
        else
        {
          std::cout << "Unable to open file: " << filename << std::endl;
          exit(1);
        }

    }

    template<typename T>
    void load_from_file ( RBM<T> * rbm
                        , std::string filename
                        , bool quiet=false
                        )
    {
        if(!quiet)
          std::cout << "loading from file:" << filename << std::endl;
        std::ifstream myfile (filename.c_str());
        if (myfile.is_open())
        {

          std::string line;
          std::string tmp;
          int stage = 0;
          bool done = false;
          while(!done&&getline(myfile,line))
          {
            if(line[0] == '#')continue;
            switch(stage)
            {
              case 0: // get v
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int v = atoi(tmp.c_str());
                if(v != rbm->v)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 1;
                break;
              }
              case 1: // get h
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int h = atoi(tmp.c_str());
                if(h != rbm->h)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 2;
                break;
              }
              case 2: // get b
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<rbm->v;j++)
                {
                    ss >> tmp;
                    rbm->b[j] = atof(tmp.c_str());
                }
                stage = 3;
                break;
              }
              case 3: // get c
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<rbm->h;j++)
                {
                    ss >> tmp;
                    rbm->c[j] = atof(tmp.c_str());
                }
                stage = 4;
                break;
              }
              case 4: // get W
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<rbm->h*rbm->v;j++)
                {
                    ss >> tmp;
                    rbm->W[j] = atof(tmp.c_str());
                }
                stage = 5;
                break;
              }
              default:done = true;break;
            }
          }

          myfile.close();
        }
        else std::cout << "Unable to open file: " << filename << std::endl;

    }

    template<typename T>
    void dump_to_file ( Perceptron<T> * perceptron
                      , std::string filename
                      , bool quiet=false
                      )
    {
        if(!quiet)
          std::cout << "dump to file:" << filename << std::endl;
        std::ofstream myfile (filename.c_str(),std::ios::out);
        if (myfile.is_open())
        {
          myfile << "#n_nodes" << std::endl;
          myfile << perceptron->n_nodes.size() << " ";
          for(int i=0;i<perceptron->n_nodes.size();i++)
          {
            myfile << perceptron->n_nodes[i] << " ";
          }
          myfile << std::endl;
          myfile << "#bias" << std::endl;
          for(int layer = 0;layer < perceptron->n_layers;layer++)
          {
            for(long i=0;i<perceptron->n_nodes[layer+1];i++)
            {
              myfile << (float)perceptron->weights_bias[layer][i] << " ";
            }
            std::cout << " ";
          }
          myfile << std::endl;
          myfile << "#weights" << std::endl;
          for(int layer = 0;layer < perceptron->n_layers;layer++)
          {
            for(int i=0;i<perceptron->n_nodes[layer+1];i++)
            {
                for(int j=0;j<perceptron->n_nodes[layer];j++)
                {
                    myfile << (float)perceptron->weights_neuron[layer][i][j] << " ";
                }
            }
          }
          myfile << std::endl;
          myfile << "#error" << std::endl;
          myfile << perceptron->final_error << std::endl;
          myfile.close();
        }
        else
        {
          std::cout << "Unable to open file: " << filename << std::endl;
          exit(1);
        }

    }

    template<typename T>
    void load_from_file ( Perceptron<T> * perceptron
                        , std::string filename
                        , bool quiet=false
                        )
    {
        if(!quiet)
          std::cout << "loading from file:" << filename << std::endl;
        std::ifstream myfile (filename.c_str());
        if (myfile.is_open())
        {
          std::string line;
          std::string tmp;
          int stage = 0;
          bool done = false;
          while(!done&&getline(myfile,line))
          {
            if(line[0] == '#')continue;
            switch(stage)
            {
              case 0: // get n_nodes
              {
                std::stringstream ss;
                ss << line;
                int n_nodes_size;
                ss >> tmp;
                n_nodes_size = atoi(tmp.c_str());
                if(n_nodes_size != perceptron->n_nodes.size())
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                for(int i=0;i<n_nodes_size;i++)
                {
                  int layer_size;
                  ss >> tmp;
                  layer_size = atoi(tmp.c_str());
                  if(layer_size != perceptron->n_nodes[i])
                  {
                    std::cout << "network structure is not consistent." << std::endl;
                    exit(1);
                  }
                }
                stage = 1;
                break;
              }
              case 1: // get bias
              {
                std::stringstream ss;
                ss << line;
                for(int layer = 0;layer < perceptron->n_layers;layer++)
                {
                  for(long i=0;i<perceptron->n_nodes[layer+1];i++)
                  {
                    ss >> tmp;
                    perceptron->weights_bias[layer][i] = atof(tmp.c_str());
                  }
                }
                stage = 2;
                break;
              }
              case 2: // get weights
              {
                std::stringstream ss;
                ss << line;
                for(int layer = 0;layer < perceptron->n_layers;layer++)
                {
                  for(int i=0;i<perceptron->n_nodes[layer+1];i++)
                  {
                      for(int j=0;j<perceptron->n_nodes[layer];j++)
                      {
                          ss >> tmp;
                          perceptron->weights_neuron[layer][i][j] = atof(tmp.c_str());
                      }
                  }
                }
                stage = 3;
                break;
              }
              case 3: // final error
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                perceptron->final_error = atof(tmp.c_str());
                stage = 4;
                break;
              }
              default:done = true;break;
            }
          }
          myfile.close();
        }
        else std::cout << "Unable to open file: " << filename << std::endl;

    }


#endif


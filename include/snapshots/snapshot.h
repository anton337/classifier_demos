#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include <fstream>
#include <iostream>

    template<typename T>
    void dump_to_file ( ConvolutionalRBM<T> * crbm
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
          myfile << crbm->v << std::endl;
          myfile << "#h" << std::endl;
          myfile << crbm->h << std::endl;
          myfile << "#nx" << std::endl;
          myfile << crbm->nx << std::endl;
          myfile << "#ny" << std::endl;
          myfile << crbm->ny << std::endl;
          myfile << "#dx" << std::endl;
          myfile << crbm->dx << std::endl;
          myfile << "#dy" << std::endl;
          myfile << crbm->dy << std::endl;
          myfile << "#kx" << std::endl;
          myfile << crbm->kx << std::endl;
          myfile << "#ky" << std::endl;
          myfile << crbm->ky << std::endl;
          myfile << "#M" << std::endl;
          myfile << crbm->M << std::endl;
          myfile << "#K (a.k.a. N)" << std::endl;
          myfile << crbm->K << std::endl;
          myfile << "#b" << std::endl;
          for(long v=0;v<crbm->M*crbm->nx*crbm->ny;v++)
          {
            myfile << crbm->b[v] << " ";
          }
          myfile << std::endl;
          myfile << "#c" << std::endl;
          for(long h=0;h<crbm->K*crbm->dx*crbm->dy;h++)
          {
            myfile << crbm->c[h] << " ";
          }
          myfile << std::endl;
          myfile << "#W" << std::endl;
          for(long k=0;k<crbm->M*crbm->K*crbm->kx*crbm->ky;k++)
          {
            myfile << crbm->W[k] << " ";
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
    void load_from_file ( ConvolutionalRBM<T> * crbm
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
                if(v != crbm->v)
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
                if(h != crbm->h)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 2;
                break;
              }
              case 2: // get nx
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int nx = atoi(tmp.c_str());
                if(nx != crbm->nx)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 3;
                break;
              }
              case 3: // get ny
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int ny = atoi(tmp.c_str());
                if(ny != crbm->ny)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 4;
                break;
              }
              case 4: // get dx
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int dx = atoi(tmp.c_str());
                if(dx != crbm->dx)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 5;
                break;
              }
              case 5: // get dy
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int dy = atoi(tmp.c_str());
                if(dy != crbm->dy)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 6;
                break;
              }
              case 6: // get kx
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int kx = atoi(tmp.c_str());
                if(kx != crbm->kx)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 7;
                break;
              }
              case 7: // get ky
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int ky = atoi(tmp.c_str());
                if(ky != crbm->ky)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 8;
                break;
              }
              case 8: // get M
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int M = atoi(tmp.c_str());
                if(M != crbm->M)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 9;
                break;
              }
              case 9: // get K (a.k.a N)
              {
                std::stringstream ss;
                ss << line;
                ss >> tmp;
                int K = atoi(tmp.c_str());
                if(K != crbm->K)
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                stage = 10;
                break;
              }
              case 10: // get b
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<crbm->M*crbm->nx*crbm->ny;j++)
                {
                    ss >> tmp;
                    crbm->b[j] = atof(tmp.c_str());
                }
                stage = 11;
                break;
              }
              case 11: // get c
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<crbm->K*crbm->dx*crbm->dy;j++)
                {
                    ss >> tmp;
                    crbm->c[j] = atof(tmp.c_str());
                }
                stage = 12;
                break;
              }
              case 12: // get W
              {
                std::stringstream ss;
                ss << line;
                for(int j=0;j<crbm->K*crbm->M*crbm->kx*crbm->ky;j++)
                {
                    ss >> tmp;
                    crbm->W[j] = atof(tmp.c_str());
                }
                stage = 13;
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
                  std::cout << "network structure is not consistent: " << n_nodes_size << " " << perceptron->n_nodes.size() << std::endl;
                  exit(1);
                }
                for(int i=0;i<n_nodes_size;i++)
                {
                  int layer_size;
                  ss >> tmp;
                  layer_size = atoi(tmp.c_str());
                  if(layer_size != perceptron->n_nodes[i])
                  {
                    std::cout << "network structure is not consistent: " << layer_size << " " << perceptron->n_nodes[i] << std::endl;
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


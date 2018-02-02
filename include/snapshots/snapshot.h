#ifndef SNAPSHOT_H
#define SNAPSHOT_H


    void dump_to_file(std::string filename,bool quiet=false)
    {
        if(!quiet)
          std::cout << "dump to file:" << filename << std::endl;
        ofstream myfile (filename.c_str());
        if (myfile.is_open())
        {
          myfile << "#n_nodes" << std::endl;
          myfile << n_nodes.size() << " ";
          for(int i=0;i<n_nodes.size();i++)
          {
            myfile << n_nodes[i] << " ";
          }
          myfile << std::endl;
          myfile << "#bias" << std::endl;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(long i=0;i<n_nodes[layer+1];i++)
            {
              myfile << (float)weights_bias[layer][i] << " ";
            }
            std::cout << " ";
          }
          myfile << std::endl;
          myfile << "#weights" << std::endl;
          for(int layer = 0;layer < n_layers;layer++)
          {
            for(int i=0;i<n_nodes[layer+1];i++)
            {
                for(int j=0;j<n_nodes[layer];j++)
                {
                    myfile << (float)weights_neuron[layer][i][j] << " ";
                }
            }
          }
          myfile << std::endl;
          myfile << "#error" << std::endl;
          myfile << final_error << std::endl;
          myfile.close();
        }
        else
        {
          cout << "Unable to open file: " << filename << std::endl;
          exit(1);
        }

    }

    void load_from_file(std::string filename,bool quiet=false)
    {
        if(!quiet)
          std::cout << "loading from file:" << filename << std::endl;
        ifstream myfile (filename.c_str());
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
                if(n_nodes_size != n_nodes.size())
                {
                  std::cout << "network structure is not consistent." << std::endl;
                  exit(1);
                }
                for(int i=0;i<n_nodes_size;i++)
                {
                  int layer_size;
                  ss >> tmp;
                  layer_size = atoi(tmp.c_str());
                  if(layer_size != n_nodes[i])
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
                for(int layer = 0;layer < n_layers;layer++)
                {
                  for(long i=0;i<n_nodes[layer+1];i++)
                  {
                    ss >> tmp;
                    weights_bias[layer][i] = atof(tmp.c_str());
                  }
                }
                stage = 2;
                break;
              }
              case 2: // get weights
              {
                std::stringstream ss;
                ss << line;
                for(int layer = 0;layer < n_layers;layer++)
                {
                  for(int i=0;i<n_nodes[layer+1];i++)
                  {
                      for(int j=0;j<n_nodes[layer];j++)
                      {
                          ss >> tmp;
                          weights_neuron[layer][i][j] = atof(tmp.c_str());
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
                final_error = atof(tmp.c_str());
                stage = 4;
                break;
              }
              default:done = true;break;
            }
          }
          myfile.close();
        }
        else cout << "Unable to open file: " << filename << std::endl;

    }


#endif


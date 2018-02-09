#ifndef MERGE_PERCEPTRONS_H
#define MERGE_PERCEPTRONS_H

#include <vector>
#include "Perceptron.h"

template<typename T>
struct MergePerceptrons
{

    Perceptron < T > * merge ( std::vector < Perceptron < T > * > in )
    {

        if(in.size()==0)
        {
            std::cout << "Vector of merged perceptrons is empty." << std::endl;
            exit(1);
        }

        long num_in = in[0]->n_nodes[0];

        long depth = in[0]->n_nodes.size();

        for(long i=0;i<in.size();i++)
        {
            if(in[i]->n_nodes[0] != num_in)
            {
                std::cout << "All merged perceptrons need to have the same number of inputs." << std::endl;
                exit(1);
            }
        }

        for(long i=0;i<in.size();i++)
        {
            if(in[i]->n_nodes.size() != depth)
            {
                std::cout << "All merged perceptrons need to have the same number of layers." << std::endl;
                exit(1);
            }
        }

        std::vector<long> nodes;

        nodes.push_back(num_in);

        for(long layer=1;layer<in[0]->n_nodes.size();layer++)
        {
            long num = 0;
            for(long i=0;i<in.size();i++)
            {
                num += in[i]->n_nodes[layer];
            }
            nodes.push_back(num);
        }

        Perceptron < T > * out = new Perceptron < T > ( nodes );

        for(long layer = 0;layer < out->n_layers;layer++)
        {
            std::cout << "layer:" << layer << std::endl;
            if(layer==0)
            {
                for(long n=0,I=0;n<in.size();n++)
                {
                    for(long i=0;i<in[n]->n_nodes[layer+1];i++,I++)
                    {
                        for(long j=0;j<in[n]->n_nodes[layer];j++)
                        {
                            out->weights_neuron[layer][I][j] = in[n]->weights_neuron[layer][i][j];
                        }
                        out->weights_bias[layer][I] = in[n]->weights_bias[layer][i];
                    }
                }
            }
            else
            {
                for(long n=0,I=0;n<in.size();n++)
                {
                    for(long i=0;i<in[n]->n_nodes[layer+1];i++,I++)
                    {
                        for(long m=0,J=0;m<in.size();m++)
                        {
                            for(long j=0;j<in[n]->n_nodes[layer];j++,J++)
                            {
                                out->weights_neuron[layer][I][J] = (n==m)?in[n]->weights_neuron[layer][i][j]:0;
                            }
                        }
                    }
                }
                for(long n=0,I=0;n<in.size();n++)
                {
                    for(long i=0;i<in[n]->n_nodes[layer+1];i++,I++)
                    {
                        out->weights_bias[layer][I] = in[n]->weights_bias[layer][i];
                    }
                }
            }
            std::cout << "done" << std::endl;
        }

        return out;

    }

};

#endif


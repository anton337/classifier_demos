#ifndef AUTO_ENCODER_CONSTRUCTOR_H
#define AUTO_ENCODER_CONSTRUCTOR_H

#include <boost/thread.hpp>
#include "Perceptron.h"
#include "BoltzmannMachine.h"
#include "ConvolutionalNN.h"
#include "ConvolutionalRBM.h"

template < typename T >
struct AutoEncoderConstructor
{

    /**************************************************************************************/
    //
    //
    //      Constructs an auto-encoder perceptron whose nn->n_layer_type are defined
    //
    //      by a list of RBMs
    //
    //      some constraints:
    //
    //          * [ [a] [b] [c] [x] [y] [z] [+] [z'] [y'] [x'] [c'] [b'] [a'] ]
    //            nn->n_layer_type have to be symmetric transpose
    //
    //
    /**************************************************************************************/
    void construct ( ConvolutionalNeuralNetwork < T > * nn 
                   , std::vector < BoltzmannMachine < T > * > const & rbms 
                   )
    {

        //////////////////////////////////////////////
        // do sanity checks                         //
        //////////////////////////////////////////////
       
        //////////////////////////////////////////////
        // check number of nn->n_layer_type ( should be odd ) //
        //////////////////////////////////////////////
        if(nn->n_layer_type.size()%2!=1)
        {
            std::cout << "Number of nn->n_layer_type should be odd." << std::endl;
        }

        //////////////////////////////////////////////
        // check if nn->n_layer_type are symmetric            //
        //////////////////////////////////////////////
        for ( long layer = 0
            ; layer < nn->n_layer_type.size()/2
            ; layer++
            )
        {
            switch(nn->n_layer_type[layer])
            {
                case FULLY_CONNECTED_LAYER :
                    {
                        std::cout << "Fully Connected" << std::endl;
                        break;
                    }
                case CONVOLUTIONAL_LAYER :
                    {
                        std::cout << "Fully Connected" << std::endl;
                        break;
                    }
                case MAX_POOLING_LAYER :
                    {
                        std::cout << "Max Pooling Layer" << std::endl;
                        break;
                    }
                case MEAN_POOLING_LAYER :
                    {
                        std::cout << "Mean Pooling Layer" << std::endl;
                        break;
                    }
                case RELU_LAYER :
                    {
                        std::cout << "ReLu Layer" << std::endl;
                        break;
                    }
                default :
                    {
                        std::cout << "Layer type not supported." << std::endl;
                        exit(0);
                    }
            }
            if(nn->n_layer_type[layer] != nn->n_layer_type[nn->n_layer_type.size()-1-layer])
            {
                std::cout << "nn->n_layer_type must be symmetric: "
                          << "check layer " << layer 
                          << std::endl;
            }
        }


        for ( long rbm = 0, layer = 0
            ; rbm < rbms.size() && layer < nn->n_layer_type.size()/2 // the other half of nn->n_layer_type is symmetric
            ; rbm++ 
            )
        {
            for ( ; ; layer++ )
            {
                switch ( nn->n_layer_type[layer] )
                {
                    case FULLY_CONNECTED_LAYER :
                        {
                            if(rbms[rbm]->type == RESTRICTED_BOLTZMANN_MACHINE_TYPE)
                            {

                                ///////////////////
                                // sanity checks //
                                ///////////////////

                                if(rbms[rbm]->h != nn->n_nodes[layer+1])
                                {
                                  std::cout << "h mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->v != nn->n_nodes[layer])
                                {
                                  std::cout << "v mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }
                                

                                //////////////////////////////////////////////////////////////////////////////////////////
                                //                                                                                      //
                                // good                                                                                 //
                                //                                                                                      //
                                // [ [] [] [P] [] [] [+] [] [] [P'] [] [] ]                                             //
                                // set up Perceptrons in symmetric locations                                            //
                                // P - forward perceptron, uses forward RBM weights                                     //
                                // P^T - reverse perceptron, uses transpose RBM weights                                 //
                                //                                                                                      //
                                //////////////////////////////////////////////////////////////////////////////////////////
                                // set up forward perceptron
                                {
                                  {
                                    for(int i=0;i<nn->nodes[layer+1];i++)
                                    {
                                      for(int j=0;j<nn->nodes[layer];j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[j*rbms[rbm]->h+i];
                                      }
                                      nn->weights_bias[layer][i] = rbms[rbm]->c[i];
                                    }
                                  }
                                }
                                // set up reverse perceptron
                                {
                                  {
                                    for(int i=0;i<nn->nodes[layer+1];i++)
                                    {
                                      for(int j=0;j<nn->nodes[layer];j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[i*rbms[rbm]->h+j];
                                      }
                                      nn->weights_bias[layer][i] = rbms[rbm]->b[i];
                                    }
                                  }
                                }
                            }
                            else
                            {
                                std::cout << "was expecting restricted boltzmann machine at layer " 
                                          << layer << std::endl;
                                exit(1);
                            }
                            rbm++;
                            break;
                        }
                    case CONVOLUTIONAL_LAYER :
                        {
                            if(rbms[rbm]->type == CONVOLUTIONAL_RESTRICTED_BOLTZMANN_MACHINE_TYPE)
                            {
                                ///////////////////
                                // sanity checks //
                                ///////////////////

                                if(rbms[rbm]->M != nn->n_features[layer])
                                {
                                  std::cout << "M mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->K != nn->n_features[layer+1])
                                {
                                  std::cout << "K mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->kx != nn->kx[layer])
                                {
                                  std::cout << "kx mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->ky != nn->ky[layer])
                                {
                                  std::cout << "ky mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->nx != nn->nx[layer])
                                {
                                  std::cout << "nx mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->ny != nn->ny[layer])
                                {
                                  std::cout << "ny mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->dx != nn->nx[layer] - 2*(nn->kx[layer]/2))
                                {
                                  std::cout << "dx mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                if(rbms[rbm]->dy != nn->ny[layer] - 2*(nn->ky[layer]/2))
                                {
                                  std::cout << "dy mismatch, layer: " << layer << std::endl;
                                  exit(1);
                                }

                                //////////////////////////////////////////////////////////////////////////////////////////
                                //                                                                                      //
                                // good                                                                                 //
                                //                                                                                      //
                                // [ [] [] [P] [] [] [+] [] [] [P'] [] [] ]                                             //
                                // set up Perceptrons in symmetric locations                                            //
                                // P - forward convolutional perceptron, uses forward convolutional RBM weights         //
                                // P^T - reverse convolutional perceptron, uses transpose convolutional RBM weights     //
                                //                                                                                      //
                                //////////////////////////////////////////////////////////////////////////////////////////
                                // set up forward convolutional perceptron
                                {
                                  {
                                    long K = rbms[rbm]->N;
                                    long M = rbms[rbm]->M;
                                    long kx = rbms[rbm]->kx;
                                    long ky = rbms[rbm]->ky;
                                    long dx = rbms[rbm]->dx;
                                    long dy = rbms[rbm]->dy;
                                    long nx = rbms[rbm]->nx;
                                    long ny = rbms[rbm]->ny;
                                    for(int z=0;z<K;z++)
                                    {
                                      for(int m=0;m<M;m++)
                                      {
                                        for(int i=0;i<ky;i++)
                                        {
                                          for(int j=0;j<kx;j++)
                                          {
                                            nn->weights_neuron[layer][K*ky+i][M*kx+j] = rbms[rbm]->W[m*K*kx*ky+z*kx*ky+i*kx+j];
                                          }
                                        }
                                      }
                                    }
                                    for(int k=0,m=0;m<M;m++)
                                    {
                                      for(int i=0;i<dy;i++)
                                      {
                                        for(int j=0;j<dx;j++,k++)
                                        {
                                          nn->weights_bias[layer][k] = rbms[rbm]->c[k];
                                        }
                                      }
                                    }
                                  }
                                }
                                // set up reverse convolutional perceptron
                                {
                                  {
                                    long K = rbms[rbm]->N;
                                    long M = rbms[rbm]->M;
                                    long kx = rbms[rbm]->kx;
                                    long ky = rbms[rbm]->ky;
                                    long dx = rbms[rbm]->dx;
                                    long dy = rbms[rbm]->dy;
                                    long nx = rbms[rbm]->nx;
                                    long ny = rbms[rbm]->ny;
                                    for(int z=0;z<K;z++)
                                    {
                                      for(int m=0;m<M;m++)
                                      {
                                        for(int i=0;i<ky;i++)
                                        {
                                          for(int j=0;j<kx;j++)
                                          {
                                            nn->weights_neuron[layer][M*kx+j][K*ky+i] = rbms[rbm]->W[m*K*kx*ky+z*kx*ky+i*kx+j];
                                          }
                                        }
                                      }
                                    }
                                    for(int z=0,k=0;z<K;z++)
                                    {
                                      for(int i=0;i<nx;i++)
                                      {
                                        for(int j=0;j<ny;j++,k++)
                                        {
                                          nn->weights_bias[layer][k] = rbms[rbm]->b[k];
                                        }
                                      }
                                    }
                                  }
                                }
                            }
                            else
                            {
                                std::cout << "was expecting convolutional restricted boltzmann machine at layer " 
                                          << layer << std::endl;
                                exit(1);
                            }
                            rbm++;
                            break;
                        }
                    case RELU_LAYER :
                        {
                            break;
                        }
                    case MAX_POOLING_LAYER :
                        {
                            break;
                        }
                    case MEAN_POOLING_LAYER :
                        {
                            break;
                        }
                    default :
                        {
                            std::cout << "Not a supported Layer Type." << std::endl;
                            exit(1);
                        }
                }
            }
        }

    }

};

#endif


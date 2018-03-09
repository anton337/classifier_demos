#ifndef AUTO_ENCODER_CONSTRUCTOR_H
#define AUTO_ENCODER_CONSTRUCTOR_H

#include "BoltzmannMachine.h"

template < typename T >
struct AutoEncoderConstructor
{

    /**************************************************************************************/
    //
    //
    //      Constructs an auto-encoder perceptron whose layers are defined
    //
    //      by a list of RBMs
    //
    //      some constraints:
    //
    //          * [ [a] [b] [c] [x] [y] [z] [+] [z'] [y'] [x'] [c'] [b'] [a'] ]
    //            layers have to be symmetric transpose
    //
    //
    /**************************************************************************************/
    void construct ( ConvolutionalNeuralNetwork < T > * nn 
                   , std::vector < LayerType > const & layers
                   , std::vector < BoltzmannMachine < T > * > const & rbms 
                   )
    {

        //////////////////////////////////////////////
        // do sanity checks                         //
        //////////////////////////////////////////////
       
        //////////////////////////////////////////////
        // check number of layers ( should be odd ) //
        //////////////////////////////////////////////
        if(layers.size()%2!=1)
        {
            std::cout << "Number of Layers should be odd." << std::endl;
        }

        //////////////////////////////////////////////
        // check if layers are symmetric            //
        //////////////////////////////////////////////
        for ( long layer = 0
            ; layer < layers.size()/2
            ; layer++
            )
        {
            switch(layers[layer])
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
            if(layers[layer] != layers[layers.size()-1-layer])
            {
                std::cout << "Layers must be symmetric: "
                          << "check layer " << layer 
                          << std::endl;
            }
        }

        for ( long rbm = 0, layer = 0
            ; rbm < rbms.size() && layer < layers.size()/2 // the other half of layers is symmetric
            ; rbm++ 
            )
        {
            for ( ; ; layer++ )
            {
                switch ( layers[layer] )
                {
                    case FULLY_CONNECTED_LAYER :
                        {
                            if(rbms[rbm]->type == RESTRICTED_BOLTZMANN_MACHINE_TYPE)
                            {
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
                                    for(int i=0;i<nodes[layer+1];i++)
                                    {
                                      for(int j=0;j<nodes[layer];j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[j*rbms[r]->h+i];
                                      }
                                      nn->weights_bias[layer][i] = rbms[rbm]->c[i];
                                    }
                                  }
                                }
                                // set up reverse perceptron
                                {
                                  {
                                    for(int i=0;i<nodes[layer+1];i++)
                                    {
                                      for(int j=0;j<nodes[layer];j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[i*rbms[r]->h+j];
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
                                    for(int i=0;i<rbms[rbm]->kx;i++)
                                    {
                                      for(int j=0;j<rbms[rbm]->ky;j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[j*rbms[r]->h+i];
                                      }
                                    }
                                    for(int i=0;i<rbms[rbm]->dx;i++)
                                    {
                                      for(int j=0;j<rbms[rbm]->dy;j++)
                                      {
                                        nn->weights_bias[layer][i] = rbms[rbm]->c[i];
                                      }
                                    }
                                  }
                                }
                                // set up reverse convolutional perceptron
                                {
                                  {
                                    for(int i=0;i<rbms[rbm]->kx;i++)
                                    {
                                      for(int j=0;j<rbms[rbm]->ky;j++)
                                      {
                                        nn->weights_neuron[layer][i][j] = rbms[rbm]->W[i*rbms[r]->h+j];
                                      }
                                    }
                                    for(int i=0;i<rbms[rbm]->nx;i++)
                                    {
                                      for(int j=0;j<rbms[rbm]->ny;j++)
                                      {
                                        nn->weights_bias[layer][i] = rbms[rbm]->b[i];
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


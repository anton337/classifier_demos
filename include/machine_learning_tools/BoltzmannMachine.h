#ifndef BOLTZMANN_MACHINE_H
#define BOLTZMANN_MACHINE_H

enum BOLTZMANN_MACHINE_TYPE
{
    UNDEFINED_TYPE                                   = 0
  , RESTRICTED_BOLTZMANN_MACHINE_TYPE                = 1
  , CONVOLUTIONAL_RESTRICTED_BOLTZMANN_MACHINE_TYPE  = 2
};

template<typename T>
struct BoltzmannMachine
{
    const BOLTZMANN_MACHINE_TYPE type; 
    BoltzmannMachine ()
        : type ( UNDEFINED_TYPE )
    {

    }

    BoltzmannMachine ( BOLTZMANN_MACHINE_TYPE p_type )
        : type ( p_type )
    {

    }
};

#endif


#include <iostream>
#include "visualization.h"
#include "readBMP.h"

int main(int argc,char ** argv)
{
    std::cout << "Running simple ann classification demo" << std::endl;

    // load input
    if(argc>0)
    {
      Image * dat = new Image();
      ImageLoad(argv[1],dat);
    }
    else
    {
      std::cout << "please specify input name [bmp format]." << std::endl;
      exit(1);
    }

    // start graphics
    startGraphics(argc,argv,"Simple ANN Classification - MNIST Digits");
    return 0;
}


#include <iostream>
#include "ConvolutionalRBM.h"
#include "readBMP.h"

void clear() 
{
  // CSI[2J clears screen, CSI[H moves the cursor to top-left corner
  std::cout << "\x1B[2J\x1B[H";
}

ConvolutionalRBM<double> * rbm = NULL;

int main(int argc,char ** argv)
{
  std::cout << "Convolutional RBM Test." << std::endl;
  srand(time(0));
  if(argc>0)
  {
    Image * img = new Image();
    ImageLoad(argv[1],img);
    long nx = 20;
    long ny = 20;
    long kx = 3;
    long ky = 3;
    long dx = nx-2*(kx/2);
    long dy = ny-2*(ky/2);
    long K = 6;
    double * dat = img->get_doubles(nx,ny);
    long n = 8;//img->get_size()/(nx*ny);
    rbm = new ConvolutionalRBM<double> ( nx*ny
                                       , K*dx*dy
                                       , nx
                                       , ny
                                       , dx
                                       , dy
                                       , kx
                                       , ky
                                       , K
                                       , n
                                       , dat
                                       );
    double eps = 1;
    bool init = true;
    double init_error;
    for(long i=0;;i++)
    {
      rbm -> init(0);
      rbm -> cd(3,eps,0);
      if(init)
      {
        init = false;
        init_error = rbm->final_error;
      }
      if(i%10==0)
      {
        clear();
        for(long z=0,k=0;z<K;z++)
        {
          for(long x=0;x<kx;x++)
          {
            for(long y=0;y<ky;y++,k++)
            {
              std::cout << rbm->W[k] << '\t';
            }
            std::cout << '\n';
          }
          std::cout << '\n';
        }
        std::cout << init_error << '\t' << rbm -> final_error << std::endl;
      }
    }
  }
  else
  {
    std::cout << "Please specify input name [bmp format]." << std::endl;
    exit(1);
  }
    std::cout << "Done!" << std::endl;
  return 0;
}


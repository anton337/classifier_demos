#include <iostream>
#include <boost/thread.hpp>
#include "visualization.h"
#include "readBMP.h"
#include "Perceptron.h"
#include "ActivationProbe.h"
#include "snapshot.h"
#include "MergePerceptrons.h"

int main(int argc,char ** argv)
{
  std::cout << "Cat Classifier" << std::endl;
  // load input
  if(argc>0)
  {
    long nx = 1600;
    long ny = 1200;
    Image * dat = new Image();
    ImageLoad(argv[1],dat);
    double * D = dat->get_doubles(nx,ny);
    VisualizeDataArray < double > * viz_in_dat = NULL;
    viz_in_dat = new VisualizeDataArray < double > ( dat->get_size()
                                                   , nx*ny
                                                   , dat->get_width()
                                                   , nx
                                                   , ny
                                                   , D
                                                   , -1 , 1 , -1 , 1
                                                   );
    addDisplay ( viz_in_dat );
  }
  else
  {
    std::cout << "Please specify input name [bmp format]." << std::endl;
    exit(1);
  }
  startGraphics(argc,argv,"Cat Classifier");
  return 0;
}

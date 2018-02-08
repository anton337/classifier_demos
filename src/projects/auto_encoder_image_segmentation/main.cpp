#include <iostream>
#include <boost/thread.hpp>
#define VISUALIZE_DATA_ARRAY
#define VISUALIZE_RBM_RECONSTRUCTION
#include "visualization.h"
#include "readBMP.h"
#include "RBM.h"
#include "snapshot.h"

std::string output_dir = "";

RBM<double> * rbm = NULL;
RBM<double> * rbm1 = NULL;
RBM<double> * rbm2 = NULL;
RBM<double> * rbm3 = NULL;
RBM<double> * rbm4 = NULL;

std::vector<RBM<double>*> rbms;

// learns to classify hand written digits
void test_rbm_image_segmentation()
{
  while(true)
  {
    double eps = 0.01;
    long n_iter = 250;
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm -> v << std::endl;
    rbm_max_layer = 1;
    for(long i=0;i<n_iter;i++)
    {
      rbm -> init(0);
      rbm -> cd(1,eps,0);
    }
    dump_to_file(rbm,output_dir+"/layer1.rbm");

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm -> h << std::endl;
    rbm_max_layer = 2;
    rbm->vis2hid(rbm->X,rbm1->X);
    for(long i=0;i<n_iter;i++)
    {
      rbm1 -> init(0);
      rbm1 -> cd(1,eps,0);
    }
    dump_to_file(rbm1,output_dir+"/layer2.rbm");

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm1 -> h << std::endl;
    rbm_max_layer = 3;
    rbm1->vis2hid(rbm1->X,rbm2->X);
    for(long i=0;i<n_iter;i++)
    {
      rbm2 -> init(0);
      rbm2 -> cd(1,eps,0);
    }
    dump_to_file(rbm2,output_dir+"/layer3.rbm");

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm2 -> h << std::endl;
    rbm_max_layer = 4;
    rbm2->vis2hid(rbm2->X,rbm3->X);
    for(long i=0;i<n_iter;i++)
    {
      rbm3 -> init(0);
      rbm3 -> cd(1,eps,0);
    }
    dump_to_file(rbm3,output_dir+"/layer4.rbm");

    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm3 -> h << std::endl;
    rbm_max_layer = 5;
    double * D4 = new double[rbm3->n*rbm3->h];
    rbm3->vis2hid(rbm3->X,D4);
    rbm4 = new RBM<double>(rbm3 -> h, 30, rbm3 -> n, D4);
    for(long i=0;i<n_iter;i++)
    {
      rbm4 -> init(0);
      rbm4 -> cd(1,eps,0);
    }
    dump_to_file(rbm4,output_dir+"/layer5.rbm");
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << rbm4 -> h << std::endl;

  }

}

int main(int argc,char ** argv)
{
    std::cout << "Running auto-encoder image segmentation demo" << std::endl;

    // load input
    if(argc>0)
    {
      Image * dat = new Image();
      ImageLoad(argv[1],dat);
      // 860 x 615
      //long nx = 20;//43;//860;//20;
      //long ny = 15;//41;//615;//15;
      //long dx = 1;
      //long dy = 1;
      long nx = 20;
      long ny = 20;
      long dx = 1;
      long dy = 1;
      double * D = dat->get_doubles(nx,ny,dx,dy);
      set_viz_data ( dat->get_size()
                   , nx*ny
                   , dat->get_width()
                   , nx
                   , ny
                   , D
                   );
      rbm = new RBM<double>(nx*ny,2000,dat->get_size()/(nx*ny),D);
      double * D1 = new double[rbm->n*rbm->h];
      rbm1 = new RBM<double>(rbm -> h, 1500, rbm -> n, D1);
      double * D2 = new double[rbm1->n*rbm1->h];
      rbm2 = new RBM<double>(rbm1 -> h, 1000, rbm1 -> n, D2);
      double * D3 = new double[rbm2->n*rbm2->h];
      rbm3 = new RBM<double>(rbm2 -> h, 750, rbm2 -> n, D3);
      double * D4 = new double[rbm3->n*rbm3->h];
      rbm4 = new RBM<double>(rbm3 -> h, 500, rbm3 -> n, D4);
      rbms.push_back(rbm);
      rbms.push_back(rbm1);
      rbms.push_back(rbm2);
      rbms.push_back(rbm3);
      rbms.push_back(rbm4);
      set_rbm_data ( rbms
                   , dat->get_size()
                   , nx*ny
                   , dat->get_width()
                   , nx
                   , ny
                   , D
                   );
    }
    else
    {
      std::cout << "Please specify input name [bmp format]." << std::endl;
      exit(1);
    }

    // load input
    if(argc>1)
    {
      output_dir = std::string(argv[2]);
      std::cout << "Output directory: " << output_dir << std::endl;
    }
    else
    {
      std::cout << "Please specify output directory." << std::endl;
      exit(1);
    }

    // run test
    new boost::thread(test_rbm_image_segmentation);

    // start graphics
    startGraphics(argc,argv,"Auto-Encoder Image Segmentation Example");
    return 0;
}


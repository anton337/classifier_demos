#include <iostream>
#include <boost/thread.hpp>
#include "sep_reader.h"
#include "best_fit_plane.h"
#define FLOAT_DISPLAY
#include "visualization.h"

std::string filename_dat = "/home/antonk/data/oxy.hdr";
std::string filename_afi = "/home/antonk/data/afi.hdr";
SEPReader reader_dat(filename_dat.c_str());
SEPReader reader_afi(filename_afi.c_str());
int ox = reader_dat.o3;
int oy = reader_dat.o2;
int oz = reader_dat.o1;
int nx = reader_dat.n3;
int ny = reader_dat.n2;
int nz = reader_dat.n1;
float * dat         = new float[nz*ny];
float * afi         = new float[nz*ny];
float * dat_flipped = new float[3*nz*ny]; // tri color
VisualizeDataArrayColor < float > * viz_in_dat = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat2 = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat3 = NULL;
VisualizeDataArrayColor < float > * viz_slice_dat4 = NULL;
int winx = 1200;
int winy = 600;
int probex = 200;
int probey = 200;
float * dat_slice_m2 = new float[3*probex*probey];
float * dat_slice_m1 = new float[3*probex*probey];
float * dat_slice_p1 = new float[3*probex*probey];
float * dat_slice_p2 = new float[3*probex*probey];

long x = 0;
bool do_load = true;

std::vector<Point<float> > pts;
BestFitPlane<float> fit;

void load()
{
  while(true)
  {
    usleep(1000);
    if(left_selected)
    {
      left_selected = false;
      float X = x;
      float Z = nz*((float)mouse_y/winy);
      float Y = ny*(2*(float)mouse_x/winx);
      std::cout << X << '\t' << Y << '\t' << Z << std::endl;
      pts.push_back(Point<float>(X,Y,Z));
      fit.init(pts);
      do_load = true;
    }
    if(right_selected)
    {
      right_selected = false;
      do_load = true;
      pts.clear();
      fit.init(pts);
    }
    if(change_pos_index)
    {
      x++;
      if(x>=nx)
      {
        x=nx-1;
      }
      do_load = true;
    }
    if(change_neg_index)
    {
      x--;
      if(x<0)
      {
        x=0;
      }
      do_load = true;
    }
    if(do_load)
    {
      do_load = false;
      reader_dat.read_sepval ( &dat[0]
                             , reader_dat.o1
                             , reader_dat.o2
                             , reader_dat.o3 + x
                             , reader_dat.n1
                             , reader_dat.n2
                             , 1 // reader.n3
                             );
      reader_afi.read_sepval ( &afi[0]
                             , reader_dat.o1
                             , reader_dat.o2
                             , reader_dat.o3 + x
                             , reader_dat.n1
                             , reader_dat.n2
                             , 1 // reader.n3
                             );
      float min_val = 100000000;
      float max_val =-100000000;
      for(long k=0;k<ny*nz;k++)
      {
        min_val = (dat[k]<min_val)?dat[k]:min_val;
        max_val = (dat[k]>max_val)?dat[k]:max_val;
      }
      float r,g,b;
      for(long y=0,k=0;y<ny;y++)
        for(long z=0;z<nz;z++,k++)
        {
          r = (dat[k]-min_val)/(max_val-min_val);
          g = r;
          b = afi[k];
          dat_flipped[ny*nz*0+(y+(nz-1-z)*ny)] = r;
          dat_flipped[ny*nz*1+(y+(nz-1-z)*ny)] = g;
          dat_flipped[ny*nz*2+(y+(nz-1-z)*ny)] = b;
          if(pts.size()>3)
          if(fit.distance(x,y,z) < 1)
          {
            dat_flipped[ny*nz*0+(y+(nz-1-z)*ny)] = 1;
          }
        }
      int W = 5;
      for(long t=0;t<pts.size();t++)
      {
        long X = pts[t].x;
        long Y = pts[t].y;
        long Z = pts[t].z;
        for(long _x=X-W;_x<=X+W;_x++)
        if(_x>=0&&_x<nx&&_x==x)
        {
          for(long y=Y-W;y<=Y+W;y++)
          for(long z=Z-W;z<=Z+W;z++)
          if(y>=0&&y<ny)
          if(z>=0&&z<nz)
          {
            dat_flipped[ny*nz*0+(y+(nz-1-z)*ny)] = 0;
            //dat_flipped[ny*nz*1+(y+(nz-1-z)*ny)] = 0;
            dat_flipped[ny*nz*2+(y+(nz-1-z)*ny)] = 0;
          }
        }
      }
      if(pts.size()>3)
      {
        for(long _x=0,k=0;_x<probex;_x++)
        for(long _y=0;_y<probey;_y++,k++)
        {
          Point<float> p = fit.get_projection(1*(_y-probey/2),1*(_x-probex/2));
          {
            long X = p.x + 10*fit.normx;
            long Y = p.y + 10*fit.normy;
            long Z = p.z;
            if(x==X&&X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
            {
              dat_slice_p2[probex*probey*0+k] = ((_x+probex)%100>5&&(_y+probey)%100>5)?dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)]:0;
              dat_slice_p2[probex*probey*1+k] = dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)];
              dat_slice_p2[probex*probey*2+k] = dat_flipped[ny*nz*2+(Y+(nz-1-Z)*ny)];
            }
          }
          {
            long X = p.x + 5*fit.normx;
            long Y = p.y + 5*fit.normy;
            long Z = p.z;
            if(x==X&&X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
            {
              dat_slice_p1[probex*probey*0+k] = ((_x+probex)%100>5&&(_y+probey)%100>5)?dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)]:0;
              dat_slice_p1[probex*probey*1+k] = dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)];
              dat_slice_p1[probex*probey*2+k] = dat_flipped[ny*nz*2+(Y+(nz-1-Z)*ny)];
            }
          }
          {
            long X = p.x - 5*fit.normx;
            long Y = p.y - 5*fit.normy;
            long Z = p.z;
            if(x==X&&X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
            {
              dat_slice_m1[probex*probey*0+k] = ((_x+probex)%100>5&&(_y+probey)%100>5)?dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)]:0;
              dat_slice_m1[probex*probey*1+k] = dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)];
              dat_slice_m1[probex*probey*2+k] = dat_flipped[ny*nz*2+(Y+(nz-1-Z)*ny)];
            }
          }
          {
            long X = p.x - 10*fit.normx;
            long Y = p.y - 10*fit.normy;
            long Z = p.z;
            if(x==X&&X>=0&&X<nx&&Y>=0&&Y<ny&&Z>=0&&Z<nz)
            {
              dat_slice_m2[probex*probey*0+k] = ((_x+probex)%100>5&&(_y+probey)%100>5)?dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)]:0;
              dat_slice_m2[probex*probey*1+k] = dat_flipped[ny*nz*1+(Y+(nz-1-Z)*ny)];
              dat_slice_m2[probex*probey*2+k] = dat_flipped[ny*nz*2+(Y+(nz-1-Z)*ny)];
            }
          }
        }
      }
    }
  }
}

int main(int argc,char ** argv)
{
    std::cout << "Fault Pick Tool." << std::endl;
    viz_in_dat = new VisualizeDataArrayColor < float > ( 3*ny*nz
                                                       , 3*ny*nz
                                                       , 3*ny*nz
                                                       , ny
                                                       , nz
                                                       , dat_flipped
                                                       , -1 , 0 , -1 , 1
                                                       );
    viz_slice_dat = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                          , 3*probex*probey
                                                          , 3*probex*probey
                                                          , probex
                                                          , probey
                                                          , dat_slice_p1
                                                          , 0 , .5 , -1 , 0
                                                          );
    viz_slice_dat2 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                           , 3*probex*probey
                                                           , 3*probex*probey
                                                           , probex
                                                           , probey
                                                           , dat_slice_m1
                                                           , .5 , 1 , -1 , 0
                                                           );
    viz_slice_dat3 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                          , 3*probex*probey
                                                          , 3*probex*probey
                                                          , probex
                                                          , probey
                                                          , dat_slice_p2
                                                          , 0 , .5 , 0 , 1
                                                          );
    viz_slice_dat4 = new VisualizeDataArrayColor < float > ( 3*probex*probey
                                                           , 3*probex*probey
                                                           , 3*probex*probey
                                                           , probex
                                                           , probey
                                                           , dat_slice_m2
                                                           , .5 , 1 , 0 , 1
                                                           );
    addDisplay ( viz_in_dat );
    addDisplay ( viz_slice_dat );
    addDisplay ( viz_slice_dat2 );
    addDisplay ( viz_slice_dat3 );
    addDisplay ( viz_slice_dat4 );
    new boost::thread(load);
    startGraphics(argc,argv,"Picker",winx,winy);
    std::cout << "Finished." << std::endl;
    return 0;
}


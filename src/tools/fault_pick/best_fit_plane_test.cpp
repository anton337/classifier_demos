#include "best_fit_plane.h"

int main()
{
  std::vector<Point<double> > pts;
  long n = 1000;
  for(long i=0;i<n;i++)
  {
    double x = rand()%100;
    double y = rand()%100;
    double z = 5*x-4*y+rand()%10;
    pts.push_back(Point<double>(x,y,z));
  }
  BestFitPlane<double> fit;
  fit.init(pts);
  return 0;
}



#ifndef BEST_FIT_PLANE_H
#define BEST_FIT_PLANE_H

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include "triangulate.h"

template<typename T>
struct BestFitPlane
{

  Triangulation<T> triangulation;

  BestFitPlane()
  {

  }

  void init(std::vector<Point<T> > const & pts)
  {
    if(pts.size()==0)
    {
      std::cout << "point set is empty." << std::endl;
      return;
    }
    anchorx = 0;
    anchory = 0;
    anchorz = 0;
    for(long k=0;k<pts.size();k++)
    {
      anchorx += pts[k].x;
      anchory += pts[k].y;
      anchorz += pts[k].z;
    }
    anchorx /= pts.size();
    anchory /= pts.size();
    anchorz /= pts.size();
    triangulation.triangulate(pts);
    triangulation.get_normal(normx,normy,normz);
    T norm = sqrt(normx*normx+normy*normy+normz*normz)+1e-5;
    normx /= norm;
    normy /= norm;
    normz /= norm;
    sidex = normy;
    sidey =-normx;
    sidez = 0;
    T side = sqrt(sidex*sidex+sidey*sidey+sidez*sidez)+1e-5;
    sidex /= side;
    sidey /= side;
    sidez /= side;
    upx = sidey*normz - sidez*normy;
    upy = sidez*normx - sidex*normz;
    upz = sidex*normy - sidey*normx;
  }

  T upx;
  T upy;
  T upz;

  T sidex;
  T sidey;
  T sidez;

  T normx;
  T normy;
  T normz;

  T anchorx;
  T anchory;
  T anchorz;

  T distance(T x,T y,T z)
  {
    return fabs((x - anchorx)*normx + (y - anchory)*normy + (z - anchorz)*normz);
  }

  Point<T> get_projection(T u,T v)
  {
    return Point<T> ( (anchorx + u*sidex + v*upx)
                    , (anchory + u*sidey + v*upy)
                    , (anchorz + u*sidez + v*upz)
                    );
  }

  bool is_inside ( Point<T> const & p )
  {
    return triangulation.is_inside(p.x,p.y,p.z,normx,normy,normz);
  }

  void get_projection ( Point<T> const & pt , Point<T> & p )
  {
    return triangulation.get_projection(pt.x,pt.y,pt.z,normx,normy,normz,p.x,p.y,p.z);
  }

};

#endif


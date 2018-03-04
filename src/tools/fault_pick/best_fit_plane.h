#ifndef BEST_FIT_PLANE_H
#define BEST_FIT_PLANE_H

#include <iostream>
#include <vector>
#include <stdlib.h>
#include <math.h>

template<typename T>
struct Point
{
  T x,y,z;
  Point(T _x,T _y,T _z)
    : x(_x), y(_y), z(_z)
  {

  }
};

template<typename T>
struct BestFitPlane
{

  T A[9];
  T b[3];
  T avg[3];

  T scalar[3];

  BestFitPlane()
  {

  }

  void init(std::vector<Point<T> > const & pts)
  {
    scalar[0] = 10;
    scalar[1] = 10;
    scalar[2] = 10;
    if(pts.size()==0)
    {
      std::cout << "point set is empty." << std::endl;
      return;
    }
    for(long i=0;i<9;i++)
    {
      A[i] = 0;
    }
    for(long i=0;i<3;i++)
    {
      b[i] = 0;
    }
    for(long i=0;i<3;i++)
    {
      avg[i] = 0;
    }
    for(long k=0;k<pts.size();k++)
    {
      avg[0] += pts[k].x/scalar[0];
      avg[1] += pts[k].y/scalar[1];
      avg[2] += pts[k].z/scalar[2];
    }
    avg[0] /= pts.size();
    avg[1] /= pts.size();
    avg[2] /= pts.size();
    T factor = 1.0 / (pts.size());
    for(long k=0;k<pts.size();k++)
    {
      A[0] += factor * (pts[k].x/scalar[0] - avg[0]) * (pts[k].x/scalar[0] - avg[0]);
      A[1] += factor * (pts[k].x/scalar[0] - avg[0]) * (pts[k].y/scalar[1] - avg[1]);
      A[2] += factor * (pts[k].x/scalar[0] - avg[0]);
      A[3] += factor * (pts[k].y/scalar[1] - avg[1]) * (pts[k].x/scalar[0] - avg[0]);
      A[4] += factor * (pts[k].y/scalar[1] - avg[1]) * (pts[k].y/scalar[1] - avg[1]);
      A[5] += factor * (pts[k].y/scalar[1] - avg[1]);
      A[6] += factor * (pts[k].x/scalar[0] - avg[0]);
      A[7] += factor * (pts[k].y/scalar[1] - avg[1]);
      A[8] += factor * 1;
      b[0] += factor * (pts[k].x/scalar[0] - avg[0]) * (pts[k].z/scalar[2] - avg[2]);
      b[1] += factor * (pts[k].y/scalar[1] - avg[1]) * (pts[k].z/scalar[2] - avg[2]);
      b[2] += factor * (pts[k].z/scalar[2] - avg[2]);
    }
    std::cout << "A\n";
    for(long x=0,k=0;x<3;x++)
    {
      for(long y=0;y<3;y++,k++)
      {
        std::cout << A[k] << '\t';
      }
      std::cout << std::endl;
    }
    std::cout << "b\n";
    for(long i=0;i<3;i++)
    {
      std::cout << b[i] << std::endl;
    }
    ////////////////////////////////////////////////////
    //                                                //
    //  A x = b; x - gives normal vector              //
    //  E = (A x - b)^2                               //
    //  /\ = dE/dx = 2A'(A x - b) = 2(H x - B)        //
    //  x := x - epsilon dE/dx                        //
    //                                                //
    ////////////////////////////////////////////////////
    T H[9];
    for(long i=0,k=0;i<3;i++)
      for(long j=0;j<3;j++,k++)
      {
        H[k] = 0;
        for(long t=0;t<3;t++)
        {
          H[k] += A[i*3+t] * A[t*3+j];
        }
      }
    T B[3];
    for(long i=0;i<3;i++)
    {
      B[i] = 0;
      for(long t=0;t<3;t++)
      {
        B[i] += A[i*3+t]*b[t];
      }
    }
    T x[3];
    for(long i=0;i<3;i++)
    {
      x[i] = 0;
    }
    T epsilon = 1e-5;
    T D[3];
    long n_iter = 1000000;
    for(long iter=0;iter<n_iter;iter++)
    {
      for(long i=0;i<3;i++)
      {
        D[i] = -B[i];
        for(long t=0;t<3;t++)
        {
          D[i] += H[i*3+t]*x[t];
        }
        x[i] -= epsilon * D[i];
      }
      if(n_iter-iter<10)
      std::cout << "normal:" << -x[0] << '\t' << -x[1] << '\t' << 1 << std::endl;
    }
    std::cout << "normal:" << -x[0] << '\t' << -x[1] << '\t' << 1 << std::endl;
    std::cout << "anchor:" << avg[0] << '\t' << avg[1] << '\t' << avg[2] << std::endl;
    normx = -x[0];
    normy = -x[1];
    normz = 1;
    T norm = sqrt(normx*normx+normy*normy+normz*normz)+1e-5;
    normx /= norm;
    normy /= norm;
    normz /= norm;
    anchorx = avg[0];
    anchory = avg[1];
    anchorz = avg[2];
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
    anchorx *= scalar[0];
    anchory *= scalar[1];
    anchorz *= scalar[2];
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

};

#endif


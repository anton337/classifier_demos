/*
 *  readBMP.h
 *
 *  Created by Nina Amenta on Sun May 23 2004.
 *  Free to good home!
 *
 */

#include <iostream>

/* Image type - contains height, width, and data */
struct Image {
    unsigned long sizeX;
    unsigned long sizeY;
    char *data;
    long size;
    long width;
    long get_size()
    {
        if(size==0)
        {
            std::cout << "call get_doubles(), size not initialized." << std::endl;
            exit(1);
        }
        return size;
    }
    long get_width()
    {
        if(width==0)
        {
            std::cout << "call get_doubles(), width not initialized." << std::endl;
            exit(1);
        }
        return width;
    }
    Image()
    {
        size = 0;
        width = 0;
    }
    double * get_doubles(int nx,int ny,int dx=1,int dy=1) 
    {
        long NX = dx*sizeX/nx-(dx-1);
        long NY = dy*sizeY/ny-(dy-1);
        width = NX;
        size = NX*NY*nx*ny;
        double * ret = new double[size];
        for(int Y=0,K=0,k=0;Y<NY;Y++)
        {
            for(int X=0;X<NX;X++,K++)
            {
                {
                    for(int y=0;y<ny;y++)
                    {
                        for(int x=0;x<nx;x++,k++)
                        {
                            ret[k] =  (double)((unsigned char)data[3*(sizeX*((NY-1-Y)*(ny/dy)+      y )+X*(nx/dx)+x)]/256.0);
                        }
                    }
                }
            }
        }
        return ret;
    }
};
typedef struct Image Image;

/* Function that reads in the image; first param is filename, second is image struct */
/* As side effect, sets w and h */
int ImageLoad(const char* filename, Image* image);


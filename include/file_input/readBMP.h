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
    double * get_doubles(int nx,int ny) 
    {
        double * ret = new double[sizeX*sizeY];
        long NX = sizeX/nx;
        long NY = sizeY/ny;
        for(int Y=0,K=0,k=0;Y<NY;Y++)
        {
            for(int X=0;X<NX;X++,K++)
            {
                {
                    for(int y=0;y<ny;y++)
                    {
                        for(int x=0;x<nx;x++,k++)
                        {
                            //if(K/(100*5)==3)
                                //std::cout << (((unsigned char)data[3*(sizeX*((NY-1-Y)*ny+(ny-1-y))+X*nx+x)]>127)?'#':' ');
                            ret[k] =  (double)((unsigned char)data[3*(sizeX*((NY-1-Y)*ny+      y )+X*nx+x)]/256.0);
                        }
                        //if(K/(100*5)==3)
                            //std::cout << std::endl;
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


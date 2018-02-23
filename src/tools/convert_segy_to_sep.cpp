#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
using namespace std;

inline float swap_endian(const float inFloat)
{
    float retVal;
    char *floatToConvert = ( char* ) & inFloat;
    char *returnFloat = ( char* ) & retVal;
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];
    return retVal;
}

void read_segy(std::string filename,std::string output)
{
    char * text_header = new char[3200];
    char * binary_header = new char[400];
    char * trace = NULL;
    long min_x = 1000000000;
    long max_x =-1000000000;
    long min_y = 1000000000;
    long max_y =-1000000000;
    long trace_len = -1;
    long nsamp = -1;
    {
        ifstream file (filename.c_str() , ios::binary);
        file.seekg (0, ios::beg); 
        file . read(text_header,3200);
        file . read(binary_header,400);
        for(int i=0;i<3200;i++)
        {
            std::cout << text_header[i];
        }
        std::cout << std::endl;
        //for(long i=0;i<400;i+=2)
        //{
        //    short nsamp = 256*binary_header[i] + binary_header[i+1];
        //    std::cout << i << " " << nsamp << std::endl;
        //}
        nsamp = 256*binary_header[20] + binary_header[21];
        std::cout << "nsamp:" << nsamp << std::endl;
        trace_len = 240 + 4*nsamp;
        trace = new char[trace_len];

        for(long tr=0;;tr++)
        {
            if(!file . read ( trace, trace_len ))
            {
                break;
            }
            //for(int i=0;i<240;i+=4)
            //{
            //    //short index = 256*(256*(256*trace[4] + trace[5]) + trace[6]) + trace[7];
            //    short index = 256*(256*(256*trace[i] + trace[i+1]) + trace[i+2]) + trace[i+3];
            //    std::cout << i << " " << index << std::endl;
            //}
            short nsamp_verify = 256*(256*(256*trace[112] + trace[113]) + trace[114]) + trace[115];
            //std::cout << "nsamp verify:" << nsamp_verify << std::endl;
            short ind_y = 256*(256*(256*trace[192] + trace[193]) + trace[194]) + trace[195];
            //std::cout << "ind_x:" << ind_x << std::endl;
            short ind_x = 256*(256*(256*trace[188] + trace[189]) + trace[190]) + trace[191];
            //std::cout << "ind_y:" << ind_y << std::endl;
            if(nsamp_verify!=nsamp)
            {
                std::cout << "nsamp doesnt match" << std::endl;
                exit(1);
            }
            min_x = (ind_x<min_x)?ind_x:min_x;
            max_x = (ind_x>max_x)?ind_x:max_x;
            min_y = (ind_y<min_y)?ind_y:min_y;
            max_y = (ind_y>max_y)?ind_y:max_y;
            if(tr%100000==0)
            {
                std::cout << min_x << " - " << max_x << " | " << min_y << " - " << max_y << std::endl;
            }
        }
        file.close();
    }

    std::cout << "second pass" << std::endl;

    long nz = nsamp;
    long nx = max_x - min_x + 1;
    long ny = max_y - min_y + 1;

    //float * dat = new float[nx*ny*nz];
    float * dat = new float[ny*nz];

    //memset(dat,0,4L*nx*ny*nz);
    memset(dat,0,4L*ny*nz);

    float min_val = 1000000000000;
    float max_val =-1000000000000;
    float tmp_val;

    long k_start = 240;

    {
        long x_prev = -2;
        long x_curr = -1;
        ifstream file (filename.c_str() , ios::binary);
        ofstream ofile (output.c_str() , ios::binary);
        file.seekg (0, ios::beg); 
        file . read(text_header,3200);
        file . read(binary_header,400);
        for(long tr=0;;tr++)
        {
            if(!file . read ( trace, trace_len ))
            {
                break;
            }
            //for(int i=0;i<240;i+=4)
            //{
            //    //short index = 256*(256*(256*trace[4] + trace[5]) + trace[6]) + trace[7];
            //    short index = 256*(256*(256*trace[i] + trace[i+1]) + trace[i+2]) + trace[i+3];
            //    std::cout << i << " " << index << std::endl;
            //}
            short nsamp_verify = 256*(256*(256*trace[112] + trace[113]) + trace[114]) + trace[115];
            //std::cout << "nsamp verify:" << nsamp_verify << std::endl;
            short ind_y = 256*(256*(256*trace[192] + trace[193]) + trace[194]) + trace[195];
            //std::cout << "ind_x:" << ind_x << std::endl;
            short ind_x = 256*(256*(256*trace[188] + trace[189]) + trace[190]) + trace[191];
            if(x_prev != x_curr&&x_prev>=0&&x_curr>=0)
            {
                ofile . write(reinterpret_cast<char*>(&dat[0]),4L*ny*nz);
                memset(dat,0,4L*ny*nz);
            }
            x_prev = x_curr;
            x_curr = ind_x;

            //std::cout << "ind_y:" << ind_y << std::endl;
            if(nsamp_verify!=nsamp)
            {
                std::cout << "nsamp doesnt match" << std::endl;
                exit(1);
            }
            long x = (ind_x - min_x);
            long y = (ind_y - min_y);
            long z = 0;
            for(long k=k_start;k<trace_len;k+=4,z++)
            {
                tmp_val = swap_endian(*reinterpret_cast<float*>(&trace[k]));
                dat[y*nz + z] = tmp_val;
                //min_val=(tmp_val<min_val)?tmp_val:min_val;
                //max_val=(tmp_val>max_val)?tmp_val:max_val;
            }
        }
        file.close();
        ofile.close();
    }


    if(trace)
    {
        delete [] trace;
    }

    std::cout << "nz:" << nz << std::endl;
    std::cout << "ny:" << ny << std::endl;
    std::cout << "nx:" << nx << std::endl;

    std::cout << "done." << std::endl;

}

int main()
{
    read_segy("/home/antonk/data/geopress.sgy","/home/antonk/data/geopress.sep");
    //read_segy("/media/antonk/C01856E31856D84C/geopress.sgy","/home/antonk/data/geopress.sep");
    //read_segy("/home/antonk/data/eagleford/dip.sgy","/home/antonk/data/eagleford/dip.sep");
    return 0;
}


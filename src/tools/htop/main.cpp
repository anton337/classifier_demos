#include <fstream>
#include <sstream>
#include <iostream>
#include <numeric>
#include <unistd.h>
#include <vector>
#include <sys/sysinfo.h>

size_t nprocs = get_nprocs();
 
std::vector<std::vector<size_t> > load_cpu_times() {
    std::ifstream proc_stat("/proc/stat");
    std::vector<std::vector<size_t> > times(nprocs+1);
    for(size_t t=0;t<times.size();t++)
    {
        std::string line;
        std::getline(proc_stat,line);
        std::stringstream ss;
        ss << line;
        std::string tmp;
        ss >> tmp; // get cpu prefix
        for (size_t time; ss >> time; times[t].push_back(time));
    }
    return times;
}
 
bool get_cpu_times(size_t &idle_time, size_t &total_time,std::vector<std::vector<size_t> > const & cpu_times,size_t &t) {
    idle_time = cpu_times[t][3];
    total_time = std::accumulate(cpu_times[t].begin(), cpu_times[t].end(), 0);
    return true;
}
 
int main(int, char *[]) {
    std::vector<size_t> previous_idle_time(nprocs+1);
    std::vector<size_t> previous_total_time(nprocs+1);
    for(size_t t=0;t<previous_idle_time.size();t++)
    {
        previous_idle_time[t] = 0;
        previous_total_time[t] = 0;
    }
    size_t W = 100;
    std::vector<std::vector<size_t> > cpu_times;
    for (size_t idle_time, total_time;; usleep(1000000)) {
        std::cout << "\x1B[2J\x1B[H";
        std::cout << "\033[" << 32 << "m";
        cpu_times = load_cpu_times();
        for(size_t i=0;i<W+2;i++)
        {
            std::cout << "~";
        }
        std::cout << std::endl;
        for(size_t t=1;t<previous_idle_time.size();t++)
        {
            get_cpu_times(idle_time, total_time, cpu_times, t);
            const float idle_time_delta = idle_time - previous_idle_time[t];
            const float total_time_delta = total_time - previous_total_time[t];
            const float utilization = (1.0 - idle_time_delta / total_time_delta);
            std::cout << '|';
            for(size_t i=0;i<W;i++)
            {
                std::cout << ((i<W*utilization)?'#':' ');
            }
            std::cout << '|';
            std::cout << '\n';
            previous_idle_time[t] = idle_time;
            previous_total_time[t] = total_time;
        }
        for(size_t i=0;i<W+2;i++)
        {
            std::cout << "~";
        }
        std::cout << std::endl;
        {
            size_t t=0;
            get_cpu_times(idle_time, total_time, cpu_times, t);
            const float idle_time_delta = idle_time - previous_idle_time[t];
            const float total_time_delta = total_time - previous_total_time[t];
            const float utilization = (1.0 - idle_time_delta / total_time_delta);
            std::cout << '|';
            for(size_t i=0;i<W;i++)
            {
                std::cout << ((i<W*utilization)?'#':' ');
            }
            std::cout << '|';
            std::cout << '\n';
            previous_idle_time[t] = idle_time;
            previous_total_time[t] = total_time;
        }
        for(size_t i=0;i<W+2;i++)
        {
            std::cout << "~";
        }
        std::cout << std::endl;
        std::cout << "\033[" << 49 << "m";
    }
    return 0;
}


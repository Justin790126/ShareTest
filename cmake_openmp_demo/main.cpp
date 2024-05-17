#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

void readCSV(std::vector<std::vector<std::string>>& result, const std::string& filename, char delimiter=',') {

    auto start = std::chrono::high_resolution_clock::now();
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    // Read lines from the file and distribute among threads
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string cell;

        vector<string> aRow(72);
        int i =0;
        while (std::getline(lineStream, cell, delimiter)) {
            aRow[i++]=std::move(cell);
            // result[omp_get_thread_num()].emplace_back(cell);
        }
        result.emplace_back(aRow);
    }
    cout << "result: " << result.size() << endl;
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
    if (file.is_open()) file.close();
}


void readCSV2(std::vector<std::vector<std::string>>& result, const std::string& filename, char delimiter=',') {

    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    int maxThreads = omp_get_max_threads();
    

    file.seekg(0, std::ios::end);

    // Use tellg to get the position of the file pointer, which is the size of the file
    unsigned long fileSize = file.tellg();
    unsigned long chunk = fileSize/maxThreads;
    
    vector<unsigned long> st(maxThreads), end(maxThreads);

    std::string line;
    file.seekg(0, std::ios::beg);
    unsigned long aRowSize = 0;
    if (std::getline(file, line)) {
        aRowSize = file.tellg();
    }
    unsigned long estimateNumOfRows = (fileSize/aRowSize);
    // cout << "estimateNumOfRows: " << estimateNumOfRows << endl;

    for (int i = 0; i < maxThreads; i++) {
        
        st[i] = i*chunk;
        end[i] = i == maxThreads - 1 ? fileSize : (i+1)*chunk;
        if (i > 0 && i < maxThreads-1) {
            file.seekg(st[i], std::ios::beg);
            if (std::getline(file, line)) {
                st[i] = file.tellg();
            }
            

            file.seekg(end[i], std::ios::beg);
            if (std::getline(file, line)) {
                end[i] = file.tellg();
            }
        }
        // cout << st[i] << "~" << end[i] << endl;
    }
    file.seekg(0, std::ios::beg);
    if (file.is_open()) file.close();

    std::vector<std::vector<std::string>> localResults[maxThreads];

    #pragma omp parallel
    {
        ifstream fp(filename);
        int threadID = omp_get_thread_num();
        // unsigned long expectSize = end[threadID+1] - st[threadID];
        unsigned long meetCnt = 0;
        
        
        localResults[threadID].resize(estimateNumOfRows/4);
        if (fp.is_open()) {
            std::string line;
            fp.seekg(st[threadID], std::ios::beg);
            while (fp.tellg() < end[threadID]) {
                std::getline(fp, line);
                std::istringstream lineStream(line);
                std::string cell;

                std::vector<std::string> aRow;
                while (std::getline(lineStream, cell, delimiter)) {
                    aRow.emplace_back(cell);
                }

                if (true) {
                    // localResults[threadID].emplace_back(aRow);
                    localResults[threadID][meetCnt++] = std::move(aRow);
                }
            }
            // 
            fp.close();
            localResults[threadID].resize(meetCnt);
        }
        
    }

    // Concatenate the local results from all threads
    for (const auto& localResult : localResults) {
        result.insert(result.end(), localResult.begin(), localResult.end());
    }
    cout << "result : " << result.size() << endl;

    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
}



void readCSV3(std::vector<std::vector<std::string>>& result, const std::string& filename, char delimiter=',') {

    auto start = std::chrono::high_resolution_clock::now();

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    int maxThreads = omp_get_max_threads();
    

    file.seekg(0, std::ios::end);

    // Use tellg to get the position of the file pointer, which is the size of the file
    unsigned long fileSize = file.tellg();
    unsigned long chunk = fileSize/maxThreads;
    
    vector<unsigned long> st(maxThreads), end(maxThreads);

    std::string line;
    file.seekg(0, std::ios::beg);
    unsigned long aRowSize = 0;
    if (std::getline(file, line)) {
        aRowSize = file.tellg();
    }
    unsigned long estimateNumOfRows = (fileSize/aRowSize);
    file.seekg(0, std::ios::beg);
    // estimateNumOfRows = 300000/2;
    // cout << "estimateNumOfRows: " << estimateNumOfRows << endl;

    
    

    result.resize(estimateNumOfRows);
    unsigned long meetCnt = 0;

    #pragma omp for nowait
    for (unsigned long i = 0; i < estimateNumOfRows; i++) {
        if (std::getline(file, line)) {
            std::istringstream lineStream(line);
            std::string cell;

            std::vector<std::string> aRow(72);
            int j = 0;
            while (std::getline(lineStream, cell, delimiter)) {
                aRow[j++]=std::move(cell);
            }
            result[meetCnt++] = std::move(aRow);
        } else {
            i = estimateNumOfRows;
        }
        
    }
    result.resize(meetCnt);

    // remain rows
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string cell;

        vector<string> aRow(72);
        int i = 0;
        while (std::getline(lineStream, cell, delimiter)) {
            aRow[i++] = std::move(cell);
            // result[omp_get_thread_num()].emplace_back(cell);
        }
        if (aRow.size() > 0) {
            result.emplace_back(aRow);
            meetCnt++;
        }
    }

    if (file.is_open()) file.close();
    cout << "meetCnt: " << meetCnt << endl;
        
    
    // Concatenate the local results from all threads
    

    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Print the execution time
    std::cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << std::endl;
}


int main() {
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
    const std::string filename = "data.csv";
    const char delimiter = ',';
    vector<vector<string>> result;

    readCSV(result, filename, delimiter);
    result.clear();
    readCSV3(result, filename, delimiter);


    return 0;
}
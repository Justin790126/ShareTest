#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
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

void method1(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) {
        #pragma omp critical
        {
            output.push_back(input[i]);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void method2(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel
    {
        std::vector<std::vector<string>> b_local;
        #pragma omp for nowait
        for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) {
            b_local.push_back(input[i]);
        }
        #pragma omp critical
        output.insert(output.end(),b_local.begin(),b_local.end()); 
    }
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void method3(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp declare reduction (merge : std::vector<vector<string>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    #pragma omp parallel for reduction(merge: output)
    for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) output.push_back(input[i]);
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void method4(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    

    #pragma omp parallel
    {
        std::vector<vector<string>> vec_private;
        #pragma omp for nowait schedule(static)
        for (unsigned long i = 0; i < (unsigned long)input.size(); i++) { 
            vec_private.push_back(input[i]);
        }
        #pragma omp for schedule(static) ordered
        for(int i=0; i<omp_get_num_threads(); i++) {
            #pragma omp ordered
            output.insert(output.end(), vec_private.begin(), vec_private.end());
        }
    }


    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void method5(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    vector<unsigned long> idxes;
    #pragma omp  for nowait
    for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) {
        // #pragma omp critical
        // cout << omp_get_thread_num() << endl;
        idxes.push_back(i);
    }

    output.resize(idxes.size());
    #pragma omp for nowait
    for (unsigned long i = 0; i < (unsigned long)idxes.size(); ++i) {
        output[i] = std::move(input[i]);
    }
    
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void method6(const vector<vector<string>>& input, vector<vector<string>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();
    vector<unsigned long> idxes;
    #pragma omp parallel
    {
        vector<unsigned long> privIdxes;
        #pragma omp  for nowait
        for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) {
            // #pragma omp critical
            // cout << omp_get_thread_num() << endl;
            privIdxes.push_back(i);
        }
        #pragma omp critical
        {
            idxes.insert(idxes.end(),privIdxes.begin(),privIdxes.end()); 
        }
        #pragma omp barrier
    }
    output.resize(idxes.size());
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (unsigned long i = 0; i < (unsigned long)idxes.size(); ++i) {
            output[i] = std::move(input[i]);
        }
        #pragma omp barrier
    }
    // cout << output.size() << endl;
    
    
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

void readCSV3(std::vector<std::vector<std::string>>& result, const std::string& filename, char delimiter=',', int colNum=6) {

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

            std::vector<std::string> aRow(colNum);
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

        vector<string> aRow(colNum);
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

std::vector<std::string> split(const std::string& s, const std::string& delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end;

    while ((end = s.find(delimiter, start)) != std::string::npos) {
        // Extract the substring from start to the position before the delimiter
        tokens.push_back(s.substr(start, end - start));

        cout << s.substr(end, delimiter.length()) << endl;
        // Move start to the position after the delimiter
        start = end + delimiter.length();
    }

    // Add the last segment (after the final delimiter or if no delimiter was found)
    tokens.push_back(s.substr(start));

    return tokens;
}

void method7(vector<vector<string>>* input, vector<vector<string>>& output, int colNum=6, string cond="")
{
    auto start = std::chrono::high_resolution_clock::now();

    // std::vector<std::string> conds = split(cond, "AND");
    std::vector<std::string> conds;
    conds.reserve(10);
    vector<string> operators;
    operators.reserve(10);
    size_t startpos = 0;
    size_t endpos;
    string s = cond;
    string delimiter = "AND";
    while ((endpos = s.find(delimiter, startpos)) != std::string::npos) {
        // Extract the substring from start to the position before the delimiter
        conds.emplace_back(s.substr(startpos, endpos - startpos));
        operators.emplace_back(s.substr(endpos, delimiter.length()));
        // Move start to the position after the delimiter
        startpos = endpos + delimiter.length();
    }
    conds.emplace_back(s.substr(startpos));
    conds.shrink_to_fit();
    operators.shrink_to_fit();

    // cout << conds.size() << ", " << operators.size() << endl;
    int csvRowNum = input->size();

    vector<vector<int>> condRes;
    condRes.resize(conds.size());
    vector<int> res(10, 0);
    for (size_t i = 0; i < conds.size(); i++)
    {
        vector<int>* boolVal = &condRes[i];
        boolVal->resize(csvRowNum, 0);
        int sum = 0;
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            #pragma omp for nowait
            for (size_t j = 0; j < 20; j++)
            {
                // sum++;
                printf("[%d] %d->%d\n", id, i, j);
            }
            
        }
        #pragma omp barrier
        
    }

    for (size_t i = 0; i < 10; i++)
    {
        cout << res[i] << ", ";
    }
    
    
    
    
    

    // vector<unsigned long> idxes;
    // #pragma omp  for nowait
    // for (unsigned long i = 0; i < (unsigned long)input.size(); ++i) {
    //     // #pragma omp critical
    //     // cout << omp_get_thread_num() << endl;
    //     idxes.push_back(i);
    // }

    // output.resize(idxes.size());
    // #pragma omp for nowait
    // for (unsigned long i = 0; i < (unsigned long)idxes.size(); ++i) {
    //     output[i] = std::move(input[i]);
    // }
    
    auto stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start)/1000.0;

    // Print the execution time
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}

int main() {
    const int N = 10;

    // vector<vector<string>> test(16000000);
    // vector<vector<string>> output;
    const std::string filename = "data.csv";
    const char delimiter = ',';
    vector<vector<string>> result;
    readCSV3(result, filename, delimiter);

    vector<vector<string>> csvRes;
    method7(&result, csvRes, 6, "(PD > 5) AND (PG < 30) AND (MIN_DIS > 5)");
    
    // method1(test, output);
    // output.clear();
    // method2(test, output);
    // output.clear();
    // method5(test, output);
    // output.clear();
    // method6(test, output);


    return 0;
}
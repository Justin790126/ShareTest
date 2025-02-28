#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

using namespace std;

class Utils
{
public:
    static int m_iVerbose;
    static Utils *GetInstance();
    Utils() {};
    ~Utils() = default;

    int isDir(const char *path);

    string GetBaseName(string input);
};

#endif /* UTILS_H */
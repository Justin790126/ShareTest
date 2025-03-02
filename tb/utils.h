#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <QVector>

using namespace std;


class ChartInfo
{
    public:
        QVector<double> m_qvdXData;
        QVector<double> m_qvdYData;
        string m_sXLabel;
        string m_sYLabel;
};

class TfTags
{
public:
    const string tagKeras = "keras";             // string type data
    const string tagEpochLoss = "epoch_loss";    // float type data
    const string tagEpochAcc = "epoch_accuracy"; // float type data
};

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
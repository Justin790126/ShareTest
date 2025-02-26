#include <QApplication>
#include "lcTensorBoard.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h> // getopt()
#include <stdio.h>
#include <getopt.h>
#include "utils.h"
using namespace std;

int main(int argc, char** argv)
{
    string logdir = "";
    struct option long_options[] = {
        {"logdir", required_argument, NULL, 'l'},
        {"verbose", required_argument, NULL, 'v'},
        {0, 0, 0, 0}  // Terminating entry
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "l:v:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'l': {
                tbArgs.m_sLogDir = string(optarg);
                break;
            }
            case 'v': {
                Utils::m_iVerbose = stoi(optarg);
                break;
            }
            default: {
                cout << "Unknown option" << endl;
                break;
            }
        }
    }


    QApplication app(argc, argv);
    LcTensorBoard tb(tbArgs);
    return app.exec();
}
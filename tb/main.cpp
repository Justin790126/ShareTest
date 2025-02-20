#include <QApplication>
#include "lcTensorBoard.h"

#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h> // getopt()
#include <stdio.h>
#include <getopt.h>

using namespace std;

int main(int argc, char** argv)
{
    string logdir = "";
    struct option long_options[] = {
        {"logdir", required_argument, NULL, 'l'},
        {0, 0, 0, 0}  // Terminating entry
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "l:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'l': {
                tbArgs.m_sLogDir = string(optarg);
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
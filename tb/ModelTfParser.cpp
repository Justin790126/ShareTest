#include "ModelTfParser.h"

ModelTfParser::ModelTfParser()
{

}

void ModelTfParser::Wait()
{
    while (isRunning())
    {
        usleep(1000);
        QApplication::processEvents();
    }
}


void ModelTfParser::run()
{
    cout << "parse tf" << endl;
}
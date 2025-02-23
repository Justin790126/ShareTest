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
    printf("Parsing log dir %s\n", m_sLogDir.c_str());
}
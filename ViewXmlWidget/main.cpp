#include <QtGui>
#include <iostream>
#include "lcXml.h"
#include "ViewXmlWafer.h"
#include "ModelXmlParser.h"

using namespace std;

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    LcXml* xml = new LcXml();

    xml->show();

    return a.exec();
}

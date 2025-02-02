#ifndef MODEL_TF_PARSER_H
#define MODEL_TF_PARSER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <QThread>
#include <QApplication>
#include <tensorflow/core/framework/summary.pb.h>
#include <tensorflow/core/util/event.pb.h>
#include <tensorflow/core/lib/io/record_reader.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/types.h>


using namespace std;
using namespace tensorflow;

class ModelTfParser : public QThread
{
    Q_OBJECT
    public:
        ModelTfParser();
        ~ModelTfParser() = default; // FIXME: resource free implementation

    protected:
        virtual void run() override;
    private:
};


#endif /* MODEL_TF_PARSER_H */
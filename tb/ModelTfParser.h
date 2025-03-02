#ifndef MODEL_TF_PARSER_H
#define MODEL_TF_PARSER_H


#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <QThread>
#include <QApplication>
#include <QDir>
#include <tensorflow/core/framework/summary.pb.h>
#include <tensorflow/core/util/event.pb.h>
#include <tensorflow/core/lib/io/record_reader.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/types.h>

#include <QVector>
#include <QDebug>

#include "utils.h"

using namespace std;
using namespace tensorflow;

/*

    Responsable for watching tf file and emit events

 */

class TfLiveInfo
{
    public:
        TfLiveInfo() {};
        ~TfLiveInfo() = default;

        string GetFileName() const { return m_sFileName; }
        void SetFileName(const string &fileName) { m_sFileName = fileName; }
        size_t GetFileSize() const { return m_sFileSize; }
        void SetFileSize(size_t fileSize) { m_sFileSize = fileSize; }
        size_t GetWatchingPos() const { return m_sWatchingPos; }
        void SetWatchingPos(size_t watchingPos) { m_sWatchingPos = watchingPos; }

        // overload cout
        friend ostream& operator<<(ostream& os, const TfLiveInfo& info) {
            os << "TfLiveInfo{" << endl;
            os << "  fileName: " << info.m_sFileName << endl;
            os << "  fileSize: " << info.m_sFileSize << endl;
            os << "  watchingPos: " << info.m_sWatchingPos << endl;
            os << "}" << endl;
            return os;
        }
    private:

        string m_sFileName;
        size_t m_sFileSize;
        size_t m_sWatchingPos;
};

class ModelTfWatcher : public QThread
{
    Q_OBJECT
    public:
        ModelTfWatcher();
        ~ModelTfWatcher(); // FIXME: resource free implementation
        void SetLogDir(const string &logDir) { m_sLogDir = logDir; }
        string GetLogDir() const { return m_sLogDir; }
        void Wait();
        void SetWatcher(bool onOff) { m_bStop = onOff; }

        vector<TfLiveInfo*>* GetLiveInfo() { return &m_vinfoTfFiles; }
        vector<string> GetSubFolder();
    signals:
        void tfFileChanged();


    protected:
        virtual void run() override;
    private:
        int m_iVerbose;
        string m_sLogDir;
        bool m_bStop = false;

        vector<string> m_vsSubdirs;
        void ListSubDir(const string & iDir, vector<string>& oSubDirs);

        vector<TfLiveInfo*> m_vinfoTfFiles;
        void ListTfFiles(const string & iDir, vector<TfLiveInfo*>& oTfFiles);

        void ClearLiveInfo();

};

/*

    ModelTfParser

 */

class ModelTfParser : public QThread
{
    Q_OBJECT
    public:

        ModelTfParser();
        ~ModelTfParser();
        void SetInputName(const string &inputName) { m_sFname = inputName; }
        string GetInputName() { return m_sFname; }

        uint64_t GetCurPos() { return m_uCurPos; }

        QVector<double>* GetEpochLoss() { return &m_qvfEpLoss; }
        QVector<double>* GetEpochAcc() { return &m_qvfEpAcc; }

        void Wait();

    protected:
        virtual void run() override;
    private:
        int m_iVerbose;
        string m_sFname;

        std::ifstream m_fp;

        uint64_t m_uCurPos;

        void ListEntry(const tensorflow::Event &event);

    private:
        void ParseKerasTag(const tensorflow::Summary::Value &value);
        void ParseFloatTensor(const tensorflow::Summary::Value &value, QVector<double>& oResult);

        QVector<double> m_qvfEpLoss;
        QVector<double> m_qvfEpAcc;
        const TfTags m_tags;
};

#endif /* MODEL_TF_PARSER_H */
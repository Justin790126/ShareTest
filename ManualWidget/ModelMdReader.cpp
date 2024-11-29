#include "ModelMdReader.h"

ModelMdReader::ModelMdReader(QObject *parent) : QThread(parent)
{
    int argc = 3;
    char *argv[3] = {"md2html", "--github", "--html-css=style.css"};
    if (initMdParser(argc, argv) != 0)
    {
        exit(1);
    }
    // char* html = process_string(test.c_str());
}

void ModelMdReader::ParseMdRec(string filename, string folder, int level, MdNode *node)
{
    std::ifstream file(filename);
    std::string line;
    std::regex link_regex(R"(\[([^\]]+)\]\(([^)]+)\))");

    int lvl = level;
    lvl += 1;
    m_iMaxLevel = std::max(m_iMaxLevel, lvl);
    string rawContent = "";
    while (std::getline(file, line))
    {
        rawContent += line + "\n";
        std::smatch match;
        std::string::const_iterator searchStart(line.cbegin());
        while (std::regex_search(searchStart, line.cend(), match, link_regex))
        {
            std::string link_text = match[1].str();
            std::string url = match[2].str();

            if (!isImageFile(url))
            {
                string fullURL = folder + "/" + url;

                std::string nxtFolder = fullURL.substr(0, fullURL.find_last_of('/'));
                // printf("Folder: %s\n", nxtFolder.c_str());
                printf("Link Text: %s, URL: %s\n", link_text.c_str(), fullURL.c_str());
                MdNode *child = new MdNode(link_text, fullURL, lvl);
                node->SetParent(node);
                node->AddChild(child);

                ParseMdRec(fullURL, nxtFolder, lvl, child);
                // ParseMdRec()
            }
            // Move to the next potential match in this line
            searchStart = match.suffix().first;
        }
    }
    char* html = process_string(rawContent.c_str());
    node->SetHtmlContent(html);
}

bool ModelMdReader::isImageFile(const std::string &url)
{
    std::string lowercaseUrl = url;
    std::transform(lowercaseUrl.begin(), lowercaseUrl.end(), lowercaseUrl.begin(), ::tolower);

    return (lowercaseUrl.find(".png") != std::string::npos ||
            lowercaseUrl.find(".jpg") != std::string::npos ||
            lowercaseUrl.find(".jpeg") != std::string::npos ||
            lowercaseUrl.find(".bmp") != std::string::npos);
}

void ModelMdReader::TraverseMdNode(MdNode *node, std::function<void(MdNode *)> callback)
{
    callback(node);
    for (MdNode *child : *node->GetChildren())
    {
        TraverseMdNode(child, callback);
    }
}

void ModelMdReader::testRun()
{
    run();
}


void ModelMdReader::Search(const std::string &key)
{
    vector<string> result;
}
void ModelMdReader::run()
{

    string folder = m_sFname.substr(0, m_sFname.find_last_of('/'));
    int level = 0;
    m_sRoot = new MdNode("Root", folder + "/README.md", level);
    ParseMdRec(m_sFname, folder, level, m_sRoot);
    // TraverseMdNode(m_sRoot, [](MdNode *node)
    //                { cout << *node << endl; });

    emit allReaded();
}
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
    std::regex src_regex("src=\"([^\"]*)\"");
    std::regex href_regex("href=\"([^\"]*)\"");
    std::regex mdimg_link_regex("!\\[(.*)\\]\\((.*)\\)");
    

    int lvl = level;
    lvl += 1;
    m_iMaxLevel = std::max(m_iMaxLevel, lvl);
    string rawContent = "";
    while (std::getline(file, line))
    {
        string newLine = line;
        // Use std::regex_replace to replace the matched src attributes
        newLine = std::regex_replace(line, src_regex, "src=\"" + folder + "/$1\"");
        newLine = std::regex_replace(newLine, href_regex, "href=\"" + folder + "/$1\"");
        newLine = std::regex_replace(newLine, mdimg_link_regex, "![$1](" + folder + "/$2)");
        

        // Parse table of contents
        std::smatch match;
        std::string::const_iterator searchStart(line.cbegin());
        while (std::regex_search(searchStart, line.cend(), match, link_regex))
        {
            std::string link_text = match[1].str();
            std::string url = match[2].str();
            string fullURL = folder + "/" + url;
            if (!isImageFile(url))
            {
                std::string nxtFolder = fullURL.substr(0, fullURL.find_last_of('/'));
                // printf("Folder: %s\n", nxtFolder.c_str());
                printf("Link Text: %s, URL: %s\n", link_text.c_str(), fullURL.c_str());
                MdNode *child = new MdNode(link_text, fullURL, lvl);
                node->SetParent(node);
                node->AddChild(child);

                ParseMdRec(fullURL, nxtFolder, lvl, child);
                // ParseMdRec()
            }
            else
            {
            
            }
            // Move to the next potential match in this line
            searchStart = match.suffix().first;
        }

        rawContent += newLine + "\n";
    }
    char *html = process_string(rawContent.c_str());
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
    m_vSearchInfos.clear();
    TraverseMdNode(m_sRoot, [this, &key](MdNode *node)
                   {
        string htmlContent = node->GetHtmlContent();
        // split htmlContent by '\n'
        int lineCount = 1;
        // split line by''
        std::istringstream iss(htmlContent);
        std::string word;
        while (iss >> word) {
            size_t found = word.find(key);
            if (found!= std::string::npos) {
                SearchInfo searchInfo(
                    node->GetKey(),
                    node->GetUrl(),
                    word,
                    lineCount);
                searchInfo.SetNode(node);
                m_vSearchInfos.push_back(searchInfo);
            }
            lineCount++;
        } });

    // print m_vSearchInfos;
    for (const SearchInfo &info : m_vSearchInfos)
    {
        cout << info << endl;
    }
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
#include "utils.h"

static Utils *utils = NULL;

int Utils::m_iVerbose = 0;

Utils *Utils::GetInstance()
{
    if (utils == NULL)
    {
        utils = new Utils();
    }
    return utils;
}

int Utils::isDir(const char *path)
{
    struct stat statbuf;

    // Use stat to get information about the file
    if (stat(path, &statbuf) != 0)
    {
        // stat failed, return -1 (error)
        perror("stat failed");
        return -1;
    }

    // Check if it's a directory
    if (S_ISDIR(statbuf.st_mode))
    {
        return 1; // It is a directory
    }
    else
    {
        return 0; // It is not a directory
    }
}
#include <iostream>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/utsname.h>

using namespace std;

// Function to set environment variable
void setEnv(const string& name, const string& value) {
    setenv(name.c_str(), value.c_str(), 1);
}

// Function to check if environment variable exists
bool envExists(const string& name) {
    return getenv(name.c_str()) != nullptr;
}

// Function to get environment variable value
string getEnv(const string& name) {
    const char* value = getenv(name.c_str());
    return value ? string(value) : "";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Error: MAXPLATFORM argument required\n";
        return 1;
    }

    // Initial environment setup
    struct utsname uts;
    uname(&uts);
    string OS = uts.sysname;
    setEnv("OS", OS);
    setEnv("INSTALL_ROOT", "/vol0/quota_ctrl/tcwup/Release/maxwell");
    setEnv("MAXWELL_VERSION", "maxwell_1.6.18.0");
    string MAXPLATFORM = argv[1];
    setEnv("MAXPLATFORM", MAXPLATFORM);
    setEnv("MAXWELL_ROOT", getEnv("INSTALL_ROOT") + "/" + getEnv("MAXWELL_VERSION"));
    setEnv("IHT", "/vol0/ap/iht/SUSE/x86_64_suse.v6b/ihtruntime");

    // Platform-specific configurations
    if (MAXPLATFORM == "metis") {
        setEnv("METIS_HOME", getEnv("MAXWELL_ROOT") + "/metis");
        setEnv("METIS_INSTALL", getEnv("METIS_HOME") + "/release");
        setEnv("METIS_PATH", getEnv("METIS_INSTALL") + "/api/release/lib");
        setEnv("METIS_LOGTOFILE", "0");
        setEnv("OPCUS_SYS", getEnv("MAXWELL_ROOT") + "/opcus_sys");
    }
    else if (MAXPLATFORM == "snps") {
        if (!envExists("PRECIM_ALIAS")) {
            cout << "undefined Synopsys_PRECIM version\n";
            cout << "Error: source " << getEnv("MAXWELL_VERSION") << " " << MAXPLATFORM << " ... failure\n";
            return 1;
        }
        cout << "Synopsys version: " << getEnv("PRECIM_ALIAS") << "\n";
    }
    else if (MAXPLATFORM == "mentor") {
        if (!envExists("MGS_ALIAS")) {
            cout << "undefined Mentor_DRA version (> ca2007)\n";
            cout << "Error: source " << getEnv("MAXWELL_VERSION") << " " << MAXPLATFORM << " ... failure\n";
            return 1;
        }
        setEnv("LITHO_TSMC_SIMULATION_API", "89390457");
        cout << "calibre_dra version: " << getEnv("MGS_ALIAS") << "\n";
    }
    else if (MAXPLATFORM == "brion") {
        if (!envExists("TFLEX_ALIAS")) {
            cout << "undenined Brion_TFlex (> rE7.1.0)\n";
            cout << "Error: source " << getEnv("MAXWELL_VERSION") << " " << MAXPLATFORM << " ... failure\n";
            return 1;
        }
        cout << "tflex_version: " << getEnv("TFLEX_ALIAS") << " .. done\n";
        setEnv("LD_PRELOAD", getEnv("IHT") + "/lib/libpython2.7.so");
        cout << getEnv("LD_PRELOAD") << "\n";
    }
    else if (MAXPLATFORM == "api_server") {
        if (!envExists("PRECIM_ALIAS") && !envExists("TFLEX_ALIAS") && !envExists("tLPC_ALIAS")) {
            cout << "support synopsys_proteus , brion_tflex and tsmc_tLPC only \n";
            cout << "Error: source " << getEnv("MAXWELL_VERSION") << " " << MAXPLATFORM << " ... failure\n";
            return 1;
        }
        if (envExists("PRECIM_ALIAS")) {
            cout << "proteus_version: " << getEnv("PRECIM_ALIAS") << " .. done\n";
        }
        else if (envExists("tLPC_ALIAS")) {
            cout << "tLPC_version: " << getEnv("tLPC_ALIAS") << " .. done\n";
        }
        else if (envExists("TFLEX_ALIAS")) {
            cout << "tflex_version: " << getEnv("TFLEX_ALIAS") << " .. done\n";
        }
    }
    else {
        cout << "undenined MAXPLATFORM \n";
        cout << "Error: source " << getEnv("MAXWELL_VERSION") << " ... failure\n";
        return 1;
    }

    // OS-specific configurations
    if (OS == "Linux") {
        if (string(uts.machine) == "x86_64") {
            string LD_LIBRARY_PATH = getEnv("LD_LIBRARY_PATH");
            if (MAXPLATFORM == "metis") {
                setEnv("MAXWELL_INSTALL", getEnv("MAXWELL_ROOT") + "/mxwl_server");
                setEnv("LD_LIBRARY_PATH", getEnv("IHT") + "/lib:" + getEnv("IHT") + "/lib64:" + LD_LIBRARY_PATH);
                setEnv("TSMC_MODEL_HOME", getEnv("METIS_INSTALL") + "/api/release/lib");
            }
            else {
                setEnv("MAXWELL_INSTALL", getEnv("MAXWELL_ROOT") + "/" + MAXPLATFORM);
                setEnv("TSMC_MODEL_HOME", getEnv("MAXWELL_INSTALL") + "/api");
                setEnv("LD_LIBRARY_PATH", getEnv("MAXWELL_INSTALL") + "/shared_libs:" + 
                       getEnv("INSTALL_ROOT") + "/share_libs/:" + 
                       getEnv("IHT") + "/lib64:" + getEnv("IHT") + "/lib:" + LD_LIBRARY_PATH);
            }

            // Check if MAXWELL_INSTALL directory exists (simplified check)
            string maxwell_install = getEnv("MAXWELL_INSTALL");
            if (access(maxwell_install.c_str(), F_OK) != 0) {
                cout << "Error: not found any " << maxwell_install << " folder... failure\n";
                return 1;
            }

            setEnv("WISDOM_HOME", getEnv("MAXWELL_ROOT") + "/wisdom");
            cout << "update maxwell_config " << getEnv("MAXWELL_VERSION") << " to system_environment successfully! \n";
            cout << "maxwell_branch " << MAXPLATFORM << " \n";
        }
        else {
            cout << "maxwell : undenined MAXPLATFORM breanch , please first contact OPC SA directly !! \n";
            return 1;
        }
    }

    return 0;
}
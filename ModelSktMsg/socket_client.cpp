#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include "ModelSktMsg.h"
#include "ModelSktClnt.h"

using namespace std;


int main() {
    // int client_socket;
    // struct sockaddr_in server_addr;
    ModelSktClnt clnt;
    if (!clnt.connect()) {
        printf("%s\n", clnt.GetStatusMsg().c_str());
        return -1;
    }
    printf("%s\n", clnt.GetStatusMsg().c_str());
    // // Send a message to the server

    ModelSktMsg msg;
    string path = "aasdfghjkjhgfdsasdfgh.qmdl";
    size_t sizePath = strlen(path.c_str())+1;
    char* cpath = new char[sizePath];
    strcpy(cpath, path.c_str());

    size_t pktLen;
    msg.serializeArr<char>(cpath, sizePath, pktLen);
    char* pkt = msg.createPkt(pktLen);


    // char* message = "Hello from the client!";
    clnt.Send(pkt, pktLen);

    // // Receive a response from the server
    // char buffer[1024];
    // memset(buffer, 0, sizeof(buffer));
    // int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
    // if (bytes_received <= 0) {
    //     std::cerr << "Error receiving from server: " << strerror(errno) << std::endl;
    // } else {
    //     std::cout << "Received from server: " << buffer << std::endl;
    // }

    clnt.Close();
    return 0;
}
#include <iostream>

#include "ModelSktMsg.h"
#include "ModelSktSvr.h"

int main() {
    ModelSktSvr svr;
    svr.init();
    svr.start();
    svr.Close();

    // Create a socket
    

    // Fork a child process to handle connections
    // pid_t child_pid = fork();
    // if (child_pid < 0) {
    //     std::cerr << "Error forking child process: " << strerror(errno) << std::endl;
    //     return 1;
    // } else if (child_pid == 0) { // Child process
    //     // Handle connections
    //     while (true) {
    //         client_addr_len = sizeof(client_addr);
    //         client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
    //         if (client_socket < 0) {
    //             std::cerr << "Error accepting connection: " << strerror(errno) << std::endl;
    //             continue;
    //         }

    //         // Handle the client connection (e.g., receive and send data)
    //         char buffer[1024];
    //         while (true) {
    //             memset(buffer, 0, sizeof(buffer));
    //             int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
    //             if (bytes_received <= 0) {
    //                 break; // Connection closed or error
    //             }

    //             std::cout << "Received from client: " << buffer << std::endl;

    //             // Send a response to the client
    //             const char* response = "Hello from the server!";
    //             send(client_socket, response, strlen(response), 0);
    //         }

    //         close(client_socket);
    //     }

    //     exit(0); // Child process exits
    // } else { // Parent process
    //     // Wait for child process to finish
    //     waitpid(child_pid, NULL, 0);
    // }

    // close(server_socket);
    return 0;
}
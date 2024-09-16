#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    int server_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_addr_len;

    // Create a socket
    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }

    // Bind the socket to a port
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(8080); // Replace with desired port

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error binding socket: " << strerror(errno) << std::endl;
        return 1;
    }

    // Listen for incoming connections
    if (listen(server_socket, 5) < 0) {
        std::cerr << "Error listening on socket: " << strerror(errno) << std::endl;
        return 1;
    }

    std::cout << "Server listening on port 8080..." << std::endl;

    // Fork a child process to handle connections
    pid_t child_pid = fork();
    if (child_pid < 0) {
        std::cerr << "Error forking child process: " << strerror(errno) << std::endl;
        return 1;
    } else if (child_pid == 0) { // Child process
        // Handle connections
        while (true) {
            client_addr_len = sizeof(client_addr);
            client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_addr_len);
            if (client_socket < 0) {
                std::cerr << "Error accepting connection: " << strerror(errno) << std::endl;
                continue;
            }

            // Handle the client connection (e.g., receive and send data)
            char buffer[1024];
            while (true) {
                memset(buffer, 0, sizeof(buffer));
                int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
                if (bytes_received <= 0) {
                    break; // Connection closed or error
                }

                std::cout << "Received from client: " << buffer << std::endl;

                // Send a response to the client
                const char* response = "Hello from the server!";
                send(client_socket, response, strlen(response), 0);
            }

            close(client_socket);
        }

        exit(0); // Child process exits
    } else { // Parent process
        // Wait for child process to finish
        waitpid(child_pid, NULL, 0);
    }

    close(server_socket);
    return 0;
}
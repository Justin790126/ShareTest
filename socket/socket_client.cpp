#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>

int main() {
    int client_socket;
    struct sockaddr_in server_addr;

    // Create a socket
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return 1;
    }

    // Connect to the server
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1"); // Replace with server IP
    server_addr.sin_port = htons(8080); // Replace with server port

    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Error connecting to server: " << strerror(errno) << std::endl;
        return 1;
    }

    std::cout << "Connected to server..." << std::endl;

    // Send a message to the server
    const char* message = "Hello from the client!";
    send(client_socket, message, strlen(message), 0);

    // Receive a response from the server
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));
    int bytes_received = recv(client_socket, buffer, sizeof(buffer), 0);
    if (bytes_received <= 0) {
        std::cerr << "Error receiving from server: " << strerror(errno) << std::endl;
    } else {
        std::cout << "Received from server: " << buffer << std::endl;
    }

    close(client_socket);
    return 0;
}
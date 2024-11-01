#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 8080
#define MAXLINE 1024

int main() {
    int server_fd, new_socket, valread;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[MAXLINE];
    char *hello = "Hello from server";

    // Creating socket file descriptor
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    // Forcefully attaching socket to the port 8080
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 10; i++) {
        // Read message from client
        valread = read(new_socket, buffer, 1024);
        printf("%s\n", buffer);

        // Prepare 1024 float numbers
        float data[1024];
        for (int j = 0; j < 1024; j++) {
            data[j] = (float)j; // Replace with your desired values
        }

        // Send data to client
        send(new_socket, data, sizeof(data), 0);
    }

    // Closing the connected socket
    close(new_socket);
    // Closing the listening socket
    shutdown(server_fd, SHUT_RDWR);
    return 0;
}
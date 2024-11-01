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
    int sockfd;
    struct sockaddr_in servaddr;

    // Creating socket file descriptor
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&servaddr, 0, sizeof(servaddr));

    // Filling server information
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);

    // Convert IPv4 and IPv6 addresses from text to binary form
    if(inet_pton(AF_INET, "127.0.0.1", &servaddr.sin_addr)<=0) {
        printf("\nInvalid address/ Address not supported \n");
        return -1;
    }

    // Connect to the server
    if (connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0) {
        perror("connection failed");
        exit(EXIT_FAILURE);
    }

    // Send a message to trigger the server
    

    // Receive the 1024 float values 10 times
    float data[1024];
    for (int i = 0; i < 10; i++) {
        char *hello = "Hello from client";
        send(sockfd, hello, strlen(hello), 0);
        printf("Hello message sent\n");

        recv(sockfd, data, sizeof(data), 0);
        printf("Received data:\n");
        for (int j = 0; j < 10; j++) {
            printf("%.2f ", data[j]);
        }
        printf("\n");
    }

    // Close the socket
    close(sockfd);
    return 0;
}
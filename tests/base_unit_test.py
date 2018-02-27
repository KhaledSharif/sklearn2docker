from unittest import TestCase


class BaseUnitTest(TestCase):
    def setUp(self):
        # generate random docker container name
        self.container_name = self.generate_random_string(15)

        # get a free TCP port for the container
        self.port = self.get_free_tcp_port()

    def tearDown(self):
        from os import system
        system("docker kill {}".format(self.container_name))

    @staticmethod
    def generate_random_string(k: int):
        from random import choice
        from string import ascii_lowercase, digits
        return ''.join(choice(ascii_lowercase + digits) for _ in range(k))

    @staticmethod
    def get_free_tcp_port():
        from socket import socket, AF_INET, SOCK_STREAM
        tcp = socket(AF_INET, SOCK_STREAM)
        tcp.bind(('', 0))
        addr, port = tcp.getsockname()
        tcp.close()
        return port


if __name__ == '__main__':
    print("This file is not a unit test; it merely contains a base from which other unit tests are built upon.")

#!/usr/bin/env python3

import serial
import re
import socket

from serial.tools import list_ports


class Tcp:
    def __init__(self, server):
        self.fd = None
        self.server = server

    def open(self):
        self.fd = socket.socket()
        self.fd.setblocking(1)
        self.fd.connect((self.server, 6666))

    def send(self, buffer):
        result = self.fd.send(buffer)
        if result == len(buffer):
            return result
        else:
            return None

    def recv(self):
        if self.fd is not None:
            buffer = self.fd.recv(65536)
            return buffer
        else:
            return None

    def close(self):
        if self.fd is not None:
            self.fd.close()


class Udp:
    def __init__(self, server=''):
        self.fd = None
        self.server = server

    def open(self):
        self.fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                socket.IPPROTO_UDP)
        self.fd.setblocking(1)
        self.fd.bind(('', 6666))

    def send(self, buffer):
        result = self.fd.sendall(buffer)
        if result == len(buffer):
            return result
        else:
            return None

    def recv(self):
        if self.fd is not None:
            buffer, remote = self.fd.recvfrom(65536)
            if self.server == '' or self.server == remote[0]:
                return buffer
            else:
                return None
        else:
            return None

    def close(self):
        if self.fd is not None:
            self.fd.close()


class Serial:
    def __init__(self):
        self.fd = None

    def open(self):
        ports = list(list_ports.comports())
        port = None
        for i in range(len(ports)):
            port = list(ports[i])
            if (re.match("USB VID:PID=303A:1001", port[2])):
                port = port[0]
                break
        if port == None:
            return None
        self.fd = serial.Serial(port, 921600, 8, 'N', 1, timeout=1)
        if self.fd.is_open:
            return self.fd
        else:
            return None

    def send(self, buffer):
        result = self.fd.write(buffer)
        if result == len(buffer):
            return result
        else:
            return None

    def recv(self):
        if self.fd.in_waiting:
            buffer = self.fd.read(self.fd.in_waiting)
            return buffer
        else:
            return None

    def close(self):
        if self.fd is not None and self.fd.is_open:
            self.fd.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午12:34
# @Author  : Guoliang PU
# @File    : basiclayer.py
# @Software: tfwrapper

class Layer(object):
    '''
    Layer is abstract class for executing special computing operations and managing nodes created by it.
    Network is composed by connected nodes(created by a special layer),In a network, only one instance of a special layer class existed.
    The relations from node A to node B, is expressed by the  layer A's outbound_nodes and layer B's inbound_nodes.
     typeoperations from inbound nodes to outbound nodes

    '''
    def __init__(self):
        self._inbound_nodes = 0

    pass

class Edge(object):
    '''
    Edge is abstract class maintain relationships between two or more layer instances(Edge in compute graph).
    When a connection occered from layer A,B to C, the instance of layer A and B is added in edge's "_inbound_layers", and the instance of layer C is added in "_outbound_layer".
    Meantime, instance of layer A and B add this node instance to their "_outbound_nodes" , instance of layer C add this node's instance to its "_inbound_nodes".
    A graph's relations are discribed by Layer's "_inbound_edges","_outbound_edges" and Edge's "_inbound_layers","_outbound_layer"
    A              B
    |              |
    ----------------
    |   Edege      |
    ----------------
            |
            C
    '''
    def __init__(self):
        self._inbound_layers = None
        self._outbound_layer = None
    pass
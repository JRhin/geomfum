"""Definition of triangle mesh."""

import csv
import numpy as np
import networkx as nx

import geomstats.backend as gs

import geomfum.backend as xgs
from geomfum.io import load_mesh
from geomfum.metric.mesh import HeatDistanceMetric
from geomfum.operator import (
    FaceDivergenceOperator,
    FaceOrientationOperator,
    FaceValuedGradient,
)

from ._base import Shape


class Graph(Shape):
    """Graph.

    Parameters
    ----------
    edge_list : array-like, shape=[n_vertices, 3]
        List of the edges present in the graph.
    """

    def __init__(
        self,
        edge_list,
    ):
        # ToDo: Change is_mesh in the Shape class
        super().__init__(is_mesh=False)

        self._edges = None
        self._vertex_signal = None

        self.metric = None

        self.graph = nx.Graph()       
        self.graph.add_edges_from(edge_list)
        
        self.vertices = gs.asarray(np.array(self.graph.nodes()))

        self._at_init()

    def _at_init(self):
        self.equip_with_operator(
            "face_valued_gradient", FaceValuedGradient.from_registry
        )
        self.equip_with_operator(
            "face_divergence", FaceDivergenceOperator.from_registry
        )
        self.equip_with_operator(
            "face_orientation_operator", FaceOrientationOperator.from_registry
        )

    @classmethod
    def from_file(
        cls,
        filename,
    ):
        """Instantiate given a file.

        Parameters
        ----------
        filename : str
            Path to the mesh file.

        Returns
        -------
        graph : Graph
            A graph.
        """
        edge_list = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                # Convert entries to integers and add as edge
                u, v = int(row[0]), int(row[1])
                edge_list.append((u, v))
        return cls(edge_list)

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.number_of_nodes()

    @property
    def edges(self):
        """Edges of the mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        if self._edges is None:
            self._edges = gs.from_numpy(np.array(self.graph.edges()))

        return self._edges

    def set_vertex_signal(self, 
                          signal):
        """Load a vertex signal.
        """
        self._vertex_signal = signal
        
        return None
    
    def equip_with_metric(self, metric):
        """Set the metric for the mesh.

        Parameters
        ----------
        metric : class
            A metric class to use for the mesh.
        """
        if metric == HeatDistanceMetric:
            self.metric = metric.from_registry(which="pp3d", shape=self)
        else:
            self.metric = metric(self)
        self._dist_matrix = None

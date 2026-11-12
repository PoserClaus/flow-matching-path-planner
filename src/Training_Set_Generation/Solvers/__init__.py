"""
Verzeichnis aller Solver.

Neuen Solver hinzufügen:
    1. Datei in Solvers/ anlegen, Klasse erbt von BaseSolver
    2. Hier importieren und ins Dictionary eintragen
"""

from .Base_Solver import BaseSolver, Solution, SolverConfig
from .AStar import AStar, AStarConfig

SOLVERS: dict[str, type[BaseSolver]] = {
    "astar": AStar,
}

"""
Exercise analysis package for form validation and movement analysis.
"""

from .squat_analyzer_base import SquatAnalyzerBase, SquatPhase, SQUAT_VIEW_ANALYZER_REGISTRY
from .squat_view_analyzers import SquatViewType, BaseSquatViewAnalyzer
from .squat_side_view import SideSquatAnalyzer
from .squat_front_view import FrontSquatAnalyzer

# Register view analyzers
SQUAT_VIEW_ANALYZER_REGISTRY = {
    'side': SideSquatAnalyzer,
    'front': FrontSquatAnalyzer
}

__all__ = [
    'SquatAnalyzerBase',
    'SquatPhase',
    'SquatViewType',
    'BaseSquatViewAnalyzer',
    'SideSquatAnalyzer',
    'FrontSquatAnalyzer',
    'SQUAT_VIEW_ANALYZER_REGISTRY'
]
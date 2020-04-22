"""
Tests of the core functionality of pipelines, sequences, and modules.
"""

from pyveg.src.pyveg_pipeline import Pipeline, Sequence, BaseModule

def test_instantiate_pipeline():
    p = Pipeline("testpipe")
    assert isinstance(p, Pipeline)
    assert p.name == "testpipe"


def test_instantiate_sequence():
    s = Sequence("testseq")
    assert isinstance(s, Sequence)
    assert s.name == "testseq"


def test_add_sequence_to_pipeline():
    p = Pipeline("testpipe")
    p += Sequence("testseq")
    assert len(p.sequences)==1
    assert p.testseq
    assert p.testseq.name == "testseq"
    assert p.testseq.parent is p

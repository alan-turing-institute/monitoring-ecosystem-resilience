"""
Tests of the core functionality of pipelines, sequences, and modules.
"""

from peep.src.peep_pipeline import BaseModule, Pipeline, Sequence


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
    assert len(p.sequences) == 1
    assert p.testseq
    assert p.testseq.name == "testseq"
    assert p.testseq.parent is p


def test_configure_sequence():
    s = Sequence("testseq")
    assert s.is_configured == False
    s.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    s.date_range = ["2001-01-01", "2020-01-01"]
    s.configure()
    assert s.is_configured == True
    assert s.output_location == "gee_532480_0174080_542720_0184320_testseq"


def test_configure_sequence_from_dict():
    s = Sequence("testseq")
    s.set_config({"collection_name": "TESTCOLL", "some_param": "TESTVAL"})
    assert s.collection_name == "TESTCOLL"
    assert s.some_param == "TESTVAL"


def test_instantiate_module():
    m = BaseModule()
    assert isinstance(m, BaseModule)
    assert m.name == "BaseModule"
    m.configure()
    # we didn't give it a name - it should take the class name
    assert m.name == "BaseModule"
    # if we do instantiate with a name, check we keep it
    m2 = BaseModule("testmod")
    m2.configure()
    assert m2.name == "testmod"


def test_add_module_to_sequence():
    s = Sequence("testseq")
    s += BaseModule()
    assert len(s.modules) == 1
    assert s.testseq_BaseModule
    assert s.testseq_BaseModule.parent is s
    assert s.testseq_BaseModule.name == "testseq_BaseModule"
    assert not s.testseq_BaseModule.is_configured


def test_configure_pipeline():
    p = Pipeline("testpipe")
    p.bounds = [532480.0, 174080.0, 542720.0, 184320.0]
    p.date_range = ["2001-01-01", "2020-01-01"]
    p.output_location = "/tmp"
    p.output_location_type = "local"
    p += Sequence("testseq")
    p.testseq += BaseModule()
    p.configure()
    assert p.testseq.testseq_BaseModule.is_configured

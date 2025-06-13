# Copyright Axelera AI, 2024

import glob
import os
from pathlib import Path
from unittest.mock import Mock, patch

from internal_tools import release_model_cards
import pytest
import strictyaml as sy
import yaml

from axelera.app import schema
from axelera.app.schema.types import Any, EmptyDict, MapPattern, Optional, Union

OP_DEFS = {
    Optional["test_operator"]: Union[EmptyDict, MapPattern[Any]],
    Optional["preprocess_operator_1"]: Union[EmptyDict, MapPattern[Any]],
    Optional["preprocess_operator_2"]: Union[EmptyDict, MapPattern[Any]],
}

NESTED = sy.MapPattern(sy.Str(), sy.Str() | sy.MapPattern(sy.Str(), sy.Str() | sy.Seq(sy.Any())))
OP = sy.MapPattern(sy.Str(), NESTED | sy.EmptyDict())
OPS = sy.Seq(OP)
INPUT = sy.MapPattern(sy.Str(), sy.Str() | OPS)


def test_get_template_stages_no_pipeline():
    yml = """\
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == []


def test_get_template_stages_empty_pipeline():
    yml = """\
        pipeline:
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == []


def test_get_template_stages_single_stage_with_template():
    yml = """\
        pipeline:
            - stage1:
                model_name: model1
                template_path: tst/template1.yml
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == [(0, 'stage1', Path('tst/template1.yml'))]


def test_get_template_stages_single_stage_no_template():
    yml = """\
        pipeline:
            - stage1:
                model_name: model1
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == [(0, 'stage1', None)]


def test_get_template_stages_multiple_stage_with_template():
    yml = """\
        pipeline:
            - stage1:
                model_name: model1
                template_path: tst/template1.yml
            - stage2:
                model_name: model1
                template_path: tst/template2.yml
            - stage3:
                model_name: model1
                template_path: tst/template3.yml
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == [
        (0, 'stage1', Path('tst/template1.yml')),
        (1, 'stage2', Path('tst/template2.yml')),
        (2, 'stage3', Path('tst/template3.yml')),
    ]


def test_get_template_stages_multiple_stage_some_with_template():
    yml = """\
        pipeline:
            - stage1:
                model_name: model1
                template_path: tst/template1.yml
            - stage2:
                model_name: model1
            - stage3:
                model_name: model1
            - stage4:
                model_name: model1
                template_path: tst/template2.yml
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    got = release_model_cards._get_template_stages(yml)
    assert got == [
        (0, 'stage1', Path('tst/template1.yml')),
        (1, 'stage2', None),
        (2, 'stage3', None),
        (3, 'stage4', Path('tst/template2.yml')),
    ]


def test_get_template_stages_template_with_env_vars():
    yml = """\
        pipeline:
            - stage1:
                model_name: model1
                template_path: ${TESTING_PATH}/template1.yml
        models:
            model1:
                class: ModelClass
                class_path: tst/model1.py
        datasets:
            dataset1:
                class: DatasetClass
                class_path: tst/dataset1.py
    """
    os.environ['TESTING_PATH'] = 'ax/testing'
    got = release_model_cards._get_template_stages(yml)
    assert got == [(0, 'stage1', Path('ax/testing/template1.yml'))]


def test_get_opname_dict():
    op = {"operator1": {"class:": "OperatorClass", "class_path": "tst/operator1.py"}}
    assert release_model_cards._get_opname(op) == "operator1"


def test_get_opname_mapping():
    op = sy.dirty_load(
        """\
    postprocess:
        - operator1:
              class: OperatorClass
              class_path: tst/operator1.py
    """
    )["postprocess"][0]
    assert isinstance(op, sy.YAML)
    assert release_model_cards._get_opname(op) == "operator1"


def test_map_update_plain():
    a = sy.dirty_load(
        """\
        a: 1
        b: two
        c: 3
        d: 4.0
    """
    )
    b = sy.dirty_load(
        """\
        e: five
        f: six
    """
    )
    release_model_cards._map_update(a, b)
    assert a.data == {"a": "1", "b": "two", "c": "3", "d": "4.0", "e": "five", "f": "six"}


def test_map_update_plain_some_fields_existing():
    a = sy.dirty_load(
        """\
        a: 1
        b: two
        c: 3
        d: 4.0
    """
    )
    b = sy.dirty_load(
        """\
        c: three
        d: 4
        e: five
        f: six
    """
    )
    release_model_cards._map_update(a, b)
    assert a.data == {"a": "1", "b": "two", "c": "3", "d": "4.0", "e": "five", "f": "six"}


def test_map_update_nested():
    a = sy.dirty_load(
        """\
        a: 1
        b:
            c: 3
            d: 4.0
    """,
        NESTED,
    )
    b = sy.dirty_load(
        """\
        b:
            e: five
            f: six
    """,
        NESTED,
    )
    release_model_cards._map_update(a, b)
    assert a.data == {
        "a": "1",
        "b": {"c": "3", "d": "4.0", "e": "five", "f": "six"},
    }


def test_map_update_mixture():
    a = sy.dirty_load(
        """\
        a: 1
        b:
            c: 3
            d: 4.0
            g:
                - 1
                - 2
                - 3
    """,
        NESTED,
    )
    b = sy.dirty_load(
        """\
        a: not used
        b:
            c: not used
            d: not used
            e: five
            f: six
            g:
                - not used
            extras:
                - a: 1
                - b: 2
    """,
        NESTED,
    )
    release_model_cards._map_update(a, b)
    assert a.data == {
        "a": "1",
        "b": {
            "c": "3",
            "d": "4.0",
            "e": "five",
            "f": "six",
            "g": ["1", "2", "3"],
            "extras": [{"a": "1"}, {"b": "2"}],
        },
    }


def test_update_ops_single():
    pipeline = sy.dirty_load(
        """\
        - operator1:
            config1: a
            config2: b
            config3: c
            eval:
                config4: d
                config5: e
    """,
        OPS,
    )
    template = sy.dirty_load(
        """\
        - operator1:
            config3: x
            config4: y
            eval:
                config4: x
                config5: y
                config6: z
                config7: zz
    """,
        OPS,
    )
    assert release_model_cards._update_ops(pipeline, template, OP).data == [
        {
            "operator1": {
                "config1": "a",
                "config2": "b",
                "config3": "c",
                "config4": "y",
                "eval": {"config4": "d", "config5": "e", "config6": "z", "config7": "zz"},
            }
        }
    ]


def test_update_ops_multiple():
    pipeline = sy.dirty_load(
        """\
        - operator1:
            config1: a
            config2: b
            config3: c
            eval:
                config4: d
                config5: e
        - operator2:
            config1: f
            config2: g
            config3: h
            eval:
                config4: i
                config5: j
    """,
        OPS,
    )
    template = sy.dirty_load(
        """\
        - operator1:
            config3: x
            config4: y
            eval:
                config4: x
                config5: y
                config6: z
                config7: zz
        - operator2:
                config1: 1
                config2: 2
                eval:
                    config4: 3
                    config6: 4
        - operator3:
            config1: True
            config2: False
    """,
        OPS,
    )

    assert release_model_cards._update_ops(pipeline, template, OP).data == [
        {
            "operator1": {
                "config1": "a",
                "config2": "b",
                "config3": "c",
                "config4": "y",
                "eval": {"config4": "d", "config5": "e", "config6": "z", "config7": "zz"},
            }
        },
        {
            "operator2": {
                "config1": "f",
                "config2": "g",
                "config3": "h",
                "eval": {"config4": "i", "config5": "j", "config6": "4"},
            }
        },
        {"operator3": {"config1": "True", "config2": "False"}},
    ]


def test_update_input_op_no_input():
    inp = None
    template = sy.dirty_load(
        """\
        config1: a
        config2: b
        config3: c
        image_processing:
            - op1:
                config1: a
                config2: b
            - op2:
    """,
        INPUT,
    )
    assert release_model_cards._update_input_op(inp, template, OP).data == {
        "config1": "a",
        "config2": "b",
        "config3": "c",
        "image_processing": [{"op1": {"config1": "a", "config2": "b"}}, {"op2": {}}],
    }


def test_update_input_op_with_input():
    inp = sy.dirty_load(
        """\
        config1: d
        config2: b
        image_processing:
            - op1:
                config1: a
            - op2:
                config1: c
    """,
        INPUT,
    )
    template = sy.dirty_load(
        """\
        config1: a
        config2: b
        config3: c
        image_processing:
            - op1:
                config1: a
                config2: b
            - op2:
    """,
        INPUT,
    )
    assert release_model_cards._update_input_op(inp, template, OP).data == {
        "config1": "d",
        "config2": "b",
        "config3": "c",
        "image_processing": [{"op1": {"config1": "a", "config2": "b"}}, {"op2": {"config1": "c"}}],
    }


def test_substitute_template():
    pipeline = """\
# TEST
# COPYRIGHT
# MESSAGE Axelera AI, 2024
axelera-model-format: 1.0.0

name: test-v7-test

description: Minimal test model YAML with some unrelated data to be preserved

internal-model-card:
  # I should be preserved
  model_card: MC-33
  model_subversion: v7.0
  model_repository: https://test.com/test/testv5
  git_commit: 1234567890qwertyuiop
  production_ML_framework: Framework # So should I
  key_metric: metric

pipeline:
  - test_stage:
      model_name: testv5s-v7-test
      # I should be preserved
      template_path: $TEST_PATH/test-template/test-test.yaml # I should go
      postprocess:
        - test_operator:
            config1: value1
            config2: value2
            config3: value3
            config4: value4
            config5: value5
"""
    template = """\
input:
  type: image
preprocess:
  - preprocess_operator_1:
      config1: ${{variable1}}
      config2: ${{variable2}}
      config3: True
  - preprocess_operator_2:
postprocess:
  - test_operator:
      config8: template_value11 # I should get added, but won't due to ruamel bug
      config9: template_value12
      config10: ${{variable3}}
      config1: template_value1
      config2: template_value2
      config3: template_value3
      config4: template_value4
      config5: template_value5
      config6: template_value6
      eval: # However this gets added
        config2: template_value7
        config3: template_value8
        config4: template_value9
        config7: template_value10


operators:
  test_operator:
    class: TestOperator
    class_path: $TEST_PATH/test_models/decoders/test.py
"""

    # Note that due to bugs in ruamel, some comments are not imported from the template.
    # This is visible in the expected output.
    # https://hitchdev.com/strictyaml/using/alpha/howto/roundtripping/
    expected = """\
# TEST
# COPYRIGHT
# MESSAGE Axelera AI, 2024

axelera-model-format: 1.0.0

name: test-v7-test

description: Minimal test model YAML with some unrelated data to be preserved

internal-model-card:
  # I should be preserved
  model_card: MC-33
  model_subversion: v7.0
  model_repository: https://test.com/test/testv5
  git_commit: 1234567890qwertyuiop
  production_ML_framework: Framework # So should I
  key_metric: metric

pipeline:
  - test_stage:
      model_name: testv5s-v7-test
      # I should be preserved
      input:
        type: image
      preprocess:
        - preprocess_operator_1:
            config1: ${{variable1}}
            config2: ${{variable2}}
            config3: True
        - preprocess_operator_2:
      postprocess:
        - test_operator:
            config1: value1
            config2: value2
            config3: value3
            config4: value4
            config5: value5
            config8: template_value11
            config9: template_value12
            config10: ${{variable3}}
            config6: template_value6
            eval: # However this gets added
              config2: template_value7
              config3: template_value8
              config4: template_value9
              config7: template_value10

operators:
  test_operator:
    class: TestOperator
    class_path: $TEST_PATH/test_models/decoders/test.py
"""
    with (
        patch.object(Path, "read_text", return_value=template),
        patch.object(schema, "generate_operators", return_value=OP_DEFS),
    ):
        merge = release_model_cards.substitute_template(pipeline, Mock())

    assert merge == expected

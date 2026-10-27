# MIT License
#
# Copyright (c) 2026 ops-tilelang contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CPU-only black-box regression tests for incremental test selection.

Each case builds a tiny source checkout and invokes the selector CLI. No
operator module, pytest plugin, accelerator runtime, or kernel is imported.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SELECTOR = Path(__file__).resolve().parents[1] / 'select_tests.py'


class SelectTestsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ops-tilelang-selector-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.changed_file = self.root / 'changed.txt'
        self.trigger_file = self.root / 'full_test_triggers.conf'
        self.trigger_file.write_text('pyproject.toml\ntests/conftest.py\n', encoding='utf-8')
        self.write('src/cann_ops_tilelang/__init__.py', 'from .ops import bar, foo\n')
        self.write('src/cann_ops_tilelang/ops/__init__.py', 'from .bar import bar\nfrom .foo import foo\n')
        self.write(
            'src/cann_ops_tilelang/ops/foo.py',
            'from .foo_kernel import kernel\n\ndef foo():\n    return kernel()\n',
        )
        self.write('src/cann_ops_tilelang/ops/foo_kernel.py', 'def kernel():\n    return 1\n')
        self.write('src/cann_ops_tilelang/ops/bar.py', 'def bar():\n    return 2\n')
        self.write('tests/__init__.py', '')
        self.write(
            'tests/test_foo.py',
            'from cann_ops_tilelang.ops.foo import foo\n\ndef test_foo():\n    assert foo() == 1\n',
        )
        self.write(
            'tests/test_bar.py',
            'from cann_ops_tilelang.ops.bar import bar\n\ndef test_bar():\n    assert bar() == 2\n',
        )

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def invoke(self, changed, *, extra_args=()):
        if changed is not None:
            self.changed_file.write_text('\n'.join(changed) + '\n', encoding='utf-8')
        return subprocess.run(
            [
                sys.executable,
                str(SELECTOR),
                '--root',
                str(self.root),
                '--changed-files',
                str(self.changed_file),
                '--full-test-triggers',
                str(self.trigger_file),
                *extra_args,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )

    def assert_selection(self, changed, mode, targets=()):
        result = self.invoke(changed)
        diagnostic = f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        self.assertEqual(result.returncode, 0, diagnostic)
        lines = result.stdout.splitlines()
        self.assertTrue(lines, diagnostic)
        self.assertEqual(lines[0], mode, diagnostic)
        self.assertEqual(set(lines[1:]), set(targets), diagnostic)
        self.assertEqual(len(lines[1:]), len(targets), 'Targets must be deduplicated.\n' + diagnostic)
        return result

    def test_absolute_import_selects_only_affected_test(self):
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'incremental', ['tests/test_foo.py'])

    def test_relative_kernel_import_is_transitive(self):
        self.assert_selection(['src/cann_ops_tilelang/ops/foo_kernel.py'], 'incremental', ['tests/test_foo.py'])

    def test_import_alias_and_module_attribute(self):
        self.write(
            'tests/test_alias.py',
            'import cann_ops_tilelang.ops.foo as foo_module\n'
            'from cann_ops_tilelang.ops.foo import foo as aliased\n'
            '\ndef test_alias():\n    assert foo_module.foo() == aliased()\n',
        )
        self.assert_selection(
            ['src/cann_ops_tilelang/ops/foo_kernel.py'],
            'incremental',
            ['tests/test_foo.py', 'tests/test_alias.py'],
        )

    def test_root_reexport_resolves_symbol_without_expanding_whole_package(self):
        self.write('tests/test_root_foo.py', 'from cann_ops_tilelang import foo\n')
        self.write('tests/test_root_bar.py', 'from cann_ops_tilelang import bar\n')
        self.assert_selection(
            ['src/cann_ops_tilelang/ops/foo_kernel.py'],
            'incremental',
            ['tests/test_foo.py', 'tests/test_root_foo.py'],
        )

    def test_root_package_alias_attribute_resolves_specific_export(self):
        self.write('tests/test_ops_foo.py', 'import cann_ops_tilelang as ops\n\ndef test_foo():\n    ops.foo()\n')
        self.write('tests/test_ops_bar.py', 'import cann_ops_tilelang as ops\n\ndef test_bar():\n    ops.bar()\n')
        self.assert_selection(
            ['src/cann_ops_tilelang/ops/foo_kernel.py'],
            'incremental',
            ['tests/test_foo.py', 'tests/test_ops_foo.py'],
        )

    def test_multiple_operators_and_tests_follow_transitive_dependency(self):
        self.write('src/cann_ops_tilelang/shared.py', 'def helper():\n    return 1\n')
        self.write(
            'src/cann_ops_tilelang/ops/foo_kernel.py',
            'from ..shared import helper\n\ndef kernel():\n    return helper()\n',
        )
        self.write(
            'src/cann_ops_tilelang/ops/bar.py',
            'from ..shared import helper\n\ndef bar():\n    return helper()\n',
        )
        self.write(
            'tests/test_both.py',
            'from cann_ops_tilelang.ops.foo import foo\nfrom cann_ops_tilelang.ops.bar import bar\n',
        )
        self.assert_selection(
            ['src/cann_ops_tilelang/shared.py'],
            'incremental',
            ['tests/test_foo.py', 'tests/test_bar.py', 'tests/test_both.py'],
        )

    def test_cross_domain_quant_dependency_reaches_attention_and_normalization(self):
        self.write('src/cann_ops_tilelang/quant/__init__.py', 'from .per_token_cast import per_token_cast\n')
        self.write(
            'src/cann_ops_tilelang/quant/per_token_cast.py',
            'from .per_token_cast_kernel import kernel\n\ndef per_token_cast():\n    return kernel()\n',
        )
        self.write('src/cann_ops_tilelang/quant/per_token_cast_kernel.py', 'def kernel():\n    return 1\n')
        self.write(
            'tests/quant/test_per_token_cast.py',
            'from cann_ops_tilelang.quant.per_token_cast import per_token_cast\n',
        )
        self.write('tests/attention/test_cache.py', 'from cann_ops_tilelang.quant import per_token_cast as cast\n')
        self.write(
            'tests/normalization/test_norm.py',
            'from cann_ops_tilelang.quant.per_token_cast import per_token_cast\n',
        )
        self.assert_selection(
            ['src/cann_ops_tilelang/quant/per_token_cast_kernel.py'],
            'incremental',
            ['tests/quant/test_per_token_cast.py', 'tests/attention/test_cache.py', 'tests/normalization/test_norm.py'],
        )

    def test_relative_test_helper_import_is_followed(self):
        self.write('tests/reference.py', 'from cann_ops_tilelang.ops.foo import foo\n')
        self.write('tests/test_reference.py', 'from .reference import foo\n')
        self.assert_selection(['tests/reference.py'], 'incremental', ['tests/test_reference.py'])
        self.assert_selection(
            ['src/cann_ops_tilelang/ops/foo_kernel.py'],
            'incremental',
            ['tests/test_foo.py', 'tests/test_reference.py'],
        )

    def test_new_operator_and_nonmatching_test_name_need_no_mapping(self):
        self.write('src/cann_ops_tilelang/new_family/__init__.py', '')
        self.write('src/cann_ops_tilelang/new_family/brand_new.py', 'def operation():\n    return 3\n')
        self.write(
            'tests/new_family/test_unrelated_name.py',
            'from cann_ops_tilelang.new_family.brand_new import operation\n',
        )
        self.assert_selection(
            ['src/cann_ops_tilelang/new_family/brand_new.py'],
            'incremental',
            ['tests/new_family/test_unrelated_name.py'],
        )

    def test_direct_test_change_runs_itself(self):
        self.assert_selection(['tests/test_foo.py'], 'incremental', ['tests/test_foo.py'])

    def test_duplicate_paths_and_dependency_routes_are_deduplicated(self):
        self.assert_selection(
            [
                'tests/test_foo.py',
                'src/cann_ops_tilelang/ops/foo.py',
                'src/cann_ops_tilelang/ops/foo_kernel.py',
                'tests/test_foo.py',
            ],
            'incremental',
            ['tests/test_foo.py'],
        )

    def test_unknown_source_runs_full_scope(self):
        self.write('src/cann_ops_tilelang/ops/unreferenced.py', 'VALUE = 1\n')
        self.assert_selection(['src/cann_ops_tilelang/ops/unreferenced.py'], 'full', ['tests'])

    def test_dynamic_local_import_runs_full_scope(self):
        self.write(
            'tests/test_dynamic.py',
            'import importlib\n\ndef test_dynamic():\n'
            '    module_name = "cann_ops_tilelang.ops.foo"\n'
            '    module = importlib.import_module(module_name)\n'
            '    module.foo()\n',
        )
        self.assert_selection(['src/cann_ops_tilelang/ops/foo_kernel.py'], 'full', ['tests'])

    def test_dynamic_import_loader_aliases_cannot_hide_affected_tests(self):
        loaders = (
            'import importlib\nload = importlib.import_module\n',
            'load = __import__\n',
            'from importlib import import_module as module_loader\nload = module_loader\n',
        )
        for loader in loaders:
            with self.subTest(loader=loader):
                self.write(
                    'tests/test_dynamic_alias.py',
                    loader + '\ndef test_dynamic():\n    load("cann_ops_tilelang.ops.foo")\n',
                )
                # The existing direct consumer must not mask this dynamic one.
                self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_package_namespace_assignment_runs_full_scope(self):
        self.write(
            'tests/test_namespace.py',
            'import cann_ops_tilelang as ops\nnamespace = ops.ops\n\ndef test_namespace():\n    namespace.foo()\n',
        )
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_eval_import_runs_full_scope(self):
        self.write(
            'tests/test_eval.py',
            'def test_eval():\n    eval("__import__(\'cann_ops_tilelang.ops.foo\')")\n',
        )
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_nested_conftest_change_runs_full_scope(self):
        self.write('tests/group/conftest.py', 'VALUE = 1\n')
        self.write('tests/group/test_implicit.py', 'def test_using_fixture(shared_fixture):\n    pass\n')
        self.assert_selection(['tests/group/conftest.py'], 'full', ['tests'])

    def test_helper_used_by_conftest_cannot_select_only_direct_consumers(self):
        self.write('tests/reference.py', 'VALUE = 1\n')
        self.write('tests/test_direct_helper.py', 'from tests.reference import VALUE\n')
        self.write('tests/group/conftest.py', 'from tests.reference import VALUE\n')
        self.write('tests/group/test_implicit.py', 'def test_using_fixture(shared_fixture):\n    pass\n')
        self.assert_selection(['tests/reference.py'], 'full', ['tests'])

    def test_plugin_helper_reaches_conftest_through_pytest_plugins(self):
        self.write('tests/reference.py', 'VALUE = 1\n')
        self.write('tests/test_direct_helper.py', 'from tests.reference import VALUE\n')
        self.write('tests/fixture_plugin.py', 'from tests.reference import VALUE\n')
        self.write('tests/group/conftest.py', 'pytest_plugins = ["tests.fixture_plugin"]\n')
        self.write('tests/group/test_implicit.py', 'def test_using_fixture(shared_fixture):\n    pass\n')
        self.assert_selection(['tests/reference.py'], 'full', ['tests'])

    def test_dynamic_pytest_plugins_runs_full_scope(self):
        self.write('tests/group/conftest.py', 'pytest_plugins = discover_plugins()\n')
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_package_star_import_runs_full_scope(self):
        self.write('tests/test_star.py', 'from cann_ops_tilelang import *\n')
        self.assert_selection(['src/cann_ops_tilelang/ops/foo_kernel.py'], 'full', ['tests'])

    def test_syntax_error_runs_full_scope(self):
        self.write('src/cann_ops_tilelang/ops/foo.py', 'def foo(:\n')
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_deleted_source_runs_full_scope(self):
        (self.root / 'src/cann_ops_tilelang/ops/foo.py').unlink()
        self.assert_selection(['src/cann_ops_tilelang/ops/foo.py'], 'full', ['tests'])

    def test_deleted_test_runs_full_scope(self):
        (self.root / 'tests/test_foo.py').unlink()
        self.assert_selection(['tests/test_foo.py'], 'full', ['tests'])

    def test_package_initializers_run_full_scope(self):
        for changed in ('src/cann_ops_tilelang/__init__.py', 'src/cann_ops_tilelang/ops/__init__.py'):
            with self.subTest(changed=changed):
                self.assert_selection([changed], 'full', ['tests'])

    def test_global_trigger_configuration_is_read(self):
        self.trigger_file.write_text('# policy comment\n\nrequirements*.txt\n', encoding='utf-8')
        self.write('requirements-dev.txt', 'pytest\n')
        self.assert_selection(['requirements-dev.txt'], 'full', ['tests'])

    def test_global_trigger_dominates_incremental_selection(self):
        self.write('pyproject.toml', '[project]\nname = "test"\n')
        self.assert_selection(['tests/test_foo.py', 'pyproject.toml'], 'full', ['tests'])

    def test_docs_only_skip(self):
        self.write('README.md', 'Documentation only.\n')
        self.write('docs/new_operator.md', 'Documentation only.\n')
        self.assert_selection(['README.md', 'docs/new_operator.md'], 'skip')

    def test_ci_only_changes_skip_operator_tests(self):
        paths = (
            '.gitcode/config/full_test_triggers.conf',
            '.gitcode/pipeline.yml',
            '.gitcode/scripts/ut.sh',
            '.gitcode/scripts/select_tests.py',
            '.gitcode/scripts/tests/test_select_tests.py',
        )
        for path in paths:
            with self.subTest(path=path):
                self.write(path, '# CI adjustment\n')
                self.assert_selection([path], 'skip')
                (self.root / path).unlink()
                self.assert_selection([path], 'skip')
        self.assert_selection([*paths, 'README.md'], 'skip')

    def test_ci_changes_preserve_source_selection(self):
        self.assert_selection(
            ['.gitcode/scripts/ut.sh', 'src/cann_ops_tilelang/ops/foo_kernel.py'],
            'incremental',
            ['tests/test_foo.py'],
        )
        self.assert_selection(['.gitcode/scripts/ut.sh', 'pyproject.toml'], 'full', ['tests'])

    def test_explicit_ci_full_test_rule_takes_precedence(self):
        self.trigger_file.write_text('pyproject.toml\n.gitcode/*\n', encoding='utf-8')
        self.assert_selection(['.gitcode/scripts/ut.sh'], 'full', ['tests'])

    def test_unknown_changes_outside_ci_still_run_full_scope(self):
        self.assert_selection(['.gitcode-other/pipeline.yml'], 'full', ['tests'])

    def test_missing_changed_file_is_error(self):
        result = self.invoke(None)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)

    def test_empty_changed_file_is_error(self):
        result = self.invoke([])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)

    def test_missing_trigger_config_is_error(self):
        self.trigger_file.unlink()
        result = self.invoke(['tests/test_foo.py'])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)

    def test_invalid_changed_path_is_error(self):
        result = self.invoke(['../outside.py'])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)

    def test_invalid_trigger_row_is_error(self):
        self.trigger_file.write_text('pyproject.toml unexpected-second-field\n', encoding='utf-8')
        result = self.invoke(['tests/test_foo.py'])
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)


if __name__ == '__main__':
    unittest.main()

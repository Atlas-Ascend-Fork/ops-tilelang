#!/usr/bin/env python3
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

"""Select affected pytest files without importing the operator package."""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Selection:
    mode: str
    targets: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Reference:
    module: str
    names: tuple[str, ...] = ()


def _internal(module: str) -> bool:
    return any(module == root or module.startswith(root + '.') for root in ('cann_ops_tilelang', 'tests'))


def _test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.startswith('tests/') and (fnmatchcase(name, 'test_*.py') or fnmatchcase(name, '*_test.py'))


def _document(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in {'.md', '.rst'} or PurePosixPath(path).name in {
        'LICENSE',
        'NOTICE',
        'COPYING',
    }


def _normalize(path: str) -> str:
    path = path.strip().replace('\\', '/')
    parsed = PurePosixPath(path)
    has_invalid_component = '..' in parsed.parts or ':' in path
    if not path or parsed.is_absolute() or has_invalid_component:
        raise ValueError(f'invalid repository-relative changed path: {path!r}')
    return str(parsed)


class DependencyGraph:
    """Track file dependencies, resolving package exports to their definitions.

    Package initializers are not expanded for every import: importing ops.foo
    depends on foo's definition, not every symbol re-exported by ops. Changes
    to initializers themselves always require the full suite.
    """

    def __init__(self, root: Path):
        self.modules: dict[str, str] = {}
        self.packages: set[str] = set()
        self.trees: dict[str, ast.Module] = {}
        self.exports: dict[str, dict[str, Reference]] = defaultdict(dict)
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.problems: list[str] = []
        for directory in ('src/cann_ops_tilelang', 'tests'):
            for file in sorted((root / directory).rglob('*.py')):
                path = file.relative_to(root).as_posix()
                parts = list(PurePosixPath(path).with_suffix('').parts)
                if parts[0] == 'src':
                    parts.pop(0)
                is_package = parts[-1] == '__init__'
                if is_package:
                    parts.pop()
                module = '.'.join(parts)
                if module in self.modules:
                    self.problems.append(f'ambiguous module {module}')
                self.modules[module] = path
                for length in range(1, len(parts) + int(is_package)):
                    self.packages.add('.'.join(parts[:length]))
                try:
                    self.trees[module] = ast.parse(file.read_text(encoding='utf-8-sig'), filename=path)
                except (OSError, UnicodeError, SyntaxError) as error:
                    self.problems.append(f'cannot parse {path}: {error}')
        # Build all export tables before resolving any import chain.
        for module, tree in self.trees.items():
            if module in self.packages:
                for node in tree.body:
                    for name, reference in self._imports(module, node):
                        self.exports[module][name] = reference
        for module, tree in self.trees.items():
            self._scan(module, tree)

    def affected_files(self, changed: str) -> set[str]:
        reverse: dict[str, set[str]] = defaultdict(set)
        for consumer, dependencies in self.edges.items():
            for dependency in dependencies:
                reverse[dependency].add(consumer)
        visited, pending = set(), [changed]
        while pending:
            path = pending.pop()
            if path not in visited:
                visited.add(path)
                pending.extend(reverse[path] - visited)
        return visited

    def _base(self, module: str, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ''
        package = module if module in self.packages else module.rpartition('.')[0]
        parts = package.split('.')
        if node.level > len(parts):
            self.problems.append(f'invalid relative import in {self.modules[module]}:{node.lineno}')
            return ''
        prefix = parts[: len(parts) - node.level + 1]
        return '.'.join(prefix + ([node.module] if node.module else []))

    def _imports(self, module: str, node: ast.AST) -> list[tuple[str, Reference]]:
        if isinstance(node, ast.Import):
            return [
                (
                    item.asname or item.name.split('.')[0],
                    Reference(item.name if item.asname else item.name.split('.')[0]),
                )
                for item in node.names
                if _internal(item.name)
            ]
        if isinstance(node, ast.ImportFrom):
            base = self._base(module, node)
            if _internal(base):
                return [(item.asname or item.name, Reference(base, (item.name,))) for item in node.names]
        return []

    def _resolve(self, reference: Reference, seen: frozenset[Reference] = frozenset()) -> str | None:
        module, names = reference.module, reference.names
        if reference in seen:
            raise ValueError(f'cyclic package export: {module}.{".".join(names)}')
        if module not in self.packages:
            if module in self.modules:
                return self.modules[module]
            raise ValueError(f'unresolved internal module: {module}')
        if not names:
            return None  # A namespace, not a dependency on every exported symbol.
        if names[0] == '*':
            raise ValueError(f'star import from internal package: {module}')
        exported = self.exports[module].get(names[0])
        child = module + '.' + names[0]
        if exported and exported != Reference(module, (names[0],)):
            return self._resolve(Reference(exported.module, exported.names + names[1:]), seen | {reference})
        if child in self.modules or child in self.packages:
            return self._resolve(Reference(child, names[1:]), seen | {reference})
        raise ValueError(f'unresolved internal export: {child}')

    def _scan(self, module: str, tree: ast.Module) -> None:
        path = self.modules[module]
        bindings: dict[str, set[Reference]] = defaultdict(set)
        loader_names = {'__import__'}
        importlib_names = {'importlib'}
        nodes = list(ast.walk(tree))
        parents = {child: parent for parent in nodes for child in ast.iter_child_nodes(parent)}

        def depend(reference: Reference, node: ast.AST) -> str | None:
            try:
                dependency = self._resolve(reference)
                if dependency:
                    self.edges[path].add(dependency)
                return dependency
            except ValueError as error:
                self.problems.append(f'{path}:{getattr(node, "lineno", 0)}: {error}')
                return None

        for node in nodes:
            for name, reference in self._imports(module, node):
                bindings[name].add(reference)
                depend(reference, node)
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == 'pytest_plugins' for target in targets):
                    try:
                        plugins = ast.literal_eval(node.value)
                        plugins = [plugins] if isinstance(plugins, str) else plugins
                        if not isinstance(plugins, (list, tuple)) or not all(isinstance(item, str) for item in plugins):
                            raise ValueError('expected constant plugin names')
                        for plugin in plugins:
                            if _internal(plugin) and depend(Reference(plugin), node) is None:
                                if plugin in self.modules:
                                    self.edges[path].add(self.modules[plugin])
                                else:
                                    self.problems.append(f'{path}:{node.lineno}: unresolved pytest plugin {plugin}')
                    except (ValueError, TypeError):
                        self.problems.append(f'{path}:{node.lineno}: dynamic pytest_plugins declaration')
            if isinstance(node, ast.Import):
                for item in node.names:
                    if _internal(item.name):
                        depend(Reference(item.name), node)
                    if item.name == 'importlib':
                        importlib_names.add(item.asname or item.name)
            elif isinstance(node, ast.ImportFrom) and node.module == 'importlib':
                loader_names.update(item.asname or item.name for item in node.names if item.name == 'import_module')

        def references(node: ast.AST) -> list[Reference]:
            attributes = []
            while isinstance(node, ast.Attribute):
                attributes.append(node.attr)
                node = node.value
            if not isinstance(node, ast.Name):
                return []
            return [Reference(ref.module, ref.names + tuple(reversed(attributes))) for ref in bindings.get(node.id, ())]

        for node in nodes:
            parent = parents.get(node)
            is_direct_call = isinstance(parent, ast.Call) and parent.func is node
            is_loader_name = isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in loader_names
            if is_loader_name and not is_direct_call:
                self.problems.append(f'{path}:{node.lineno}: dynamic import function used as a value')
            if isinstance(node, ast.Attribute):
                is_import_module = (
                    node.attr == 'import_module'
                    and isinstance(node.value, ast.Name)
                    and node.value.id in importlib_names
                )
                if is_import_module and not is_direct_call:
                    self.problems.append(f'{path}:{node.lineno}: dynamic import function used as a value')
                if not isinstance(parent, ast.Attribute) or parent.value is not node:
                    for reference in references(node):
                        if depend(reference, node) is None:
                            self.problems.append(f'{path}:{node.lineno}: internal package used dynamically')
            elif isinstance(node, ast.Call):
                function = node.func
                loader = isinstance(function, ast.Name) and function.id in loader_names
                loader = loader or (
                    isinstance(function, ast.Attribute)
                    and function.attr == 'import_module'
                    and isinstance(function.value, ast.Name)
                    and function.value.id in importlib_names
                )
                if loader:
                    name = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
                    if not isinstance(name, str) or _internal(name) or name.startswith('.'):
                        self.problems.append(f'{path}:{node.lineno}: dynamic internal import')
                if (
                    isinstance(function, ast.Name)
                    and function.id in {'getattr', 'hasattr', 'setattr', 'delattr'}
                    and node.args
                ):
                    refs = references(node.args[0])
                    attribute = (
                        node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else None
                    )
                    for reference in refs:
                        if isinstance(attribute, str) and function.id in {'getattr', 'hasattr'}:
                            if depend(Reference(reference.module, reference.names + (attribute,)), node) is None:
                                self.problems.append(f'{path}:{node.lineno}: internal package used dynamically')
                        else:
                            self.problems.append(f'{path}:{node.lineno}: dynamic internal attribute access')
                if isinstance(function, ast.Name) and function.id in {'eval', 'exec', 'globals', 'locals'}:
                    self.problems.append(f'{path}:{node.lineno}: dynamic namespace access')
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bindings:
                parent = parents.get(node)
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue
                is_attribute_access = (
                    isinstance(parent, ast.Call)
                    and isinstance(parent.func, ast.Name)
                    and parent.func.id in {'getattr', 'hasattr', 'setattr', 'delattr'}
                )
                if is_attribute_access and parent.args and parent.args[0] is node:
                    continue
                for reference in bindings[node.id]:
                    if depend(reference, node) is None:
                        self.problems.append(f'{path}:{node.lineno}: internal package used dynamically')


def select_tests(root: Path, changed_paths: list[str], full_patterns: list[str]) -> Selection:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'repository root does not exist: {root}')
    if not changed_paths:
        raise ValueError('changed-file list is empty')
    paths = sorted({_normalize(path) for path in changed_paths})

    def full(reason: str) -> Selection:
        return Selection('full', ('tests',), (reason,))

    code = []
    for path in paths:
        for pattern in full_patterns:
            if fnmatchcase(path, pattern):
                return full(f'{path}: full-test rule {pattern}')
        # Keep CI-only adjustments available without forcing operator tests.
        # Explicit full-test rules above still take precedence.
        if _document(path) or path.startswith('.gitcode/'):
            continue
        if not path.endswith('.py') or not path.startswith(('src/cann_ops_tilelang/', 'tests/')):
            return full(f'{path}: unclassified non-document change')
        if not (root / path).is_file():
            return full(f'{path}: deleted or unavailable Python file')
        if PurePosixPath(path).name == '__init__.py':
            return full(f'{path}: package initializer changed')
        code.append(path)
    if not code:
        return Selection('skip', (), ('only documentation or CI files changed',))

    graph = DependencyGraph(root)
    if graph.problems:
        return full('dependency analysis is uncertain: ' + graph.problems[0])
    targets, reasons = set(), []
    for path in code:
        affected_files = graph.affected_files(path)
        if any(PurePosixPath(consumer).name == 'conftest.py' for consumer in affected_files):
            return full(f'{path}: affects implicitly loaded pytest fixtures')
        affected = {consumer for consumer in affected_files if _test_file(consumer)}
        if not affected:
            return full(f'{path}: no affected tests could be established')
        targets.update(affected)
        reasons.append(f'{path} -> {", ".join(sorted(affected))}')
    return Selection('incremental', tuple(sorted(targets)), tuple(reasons))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--changed-files', type=Path, required=True)
    parser.add_argument('--full-test-triggers', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        changed = [
            line.strip()
            for line in arguments.changed_files.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()
        ]
        patterns = [
            line.strip()
            for line in arguments.full_test_triggers.read_text(encoding='utf-8-sig').splitlines()
            if line.strip() and not line.lstrip().startswith('#')
        ]
        if not patterns or any(len(pattern.split()) != 1 for pattern in patterns):
            raise ValueError('full-test trigger config must contain one nonempty pattern per rule')
        result = select_tests(arguments.root, changed, patterns)
    except (OSError, ValueError) as error:
        LOGGER.error('test selection failed: %s', error)
        return 2
    for reason in result.reasons:
        LOGGER.info('%s', reason)
    # stdout is the machine-readable protocol consumed by ut.sh.
    sys.stdout.write('\n'.join((result.mode, *result.targets)) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

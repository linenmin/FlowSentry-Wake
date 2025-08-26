#!/usr/bin/env python3
# Copyright Axelera AI, 2025

from collections.abc import MutableMapping
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from typing import Dict

import yaml

"""
This module provides support functions for the installer scripts.
It is very much WIP and is expected to grow and change considerably as
existing bash code in the installer scripts is ported to python, and extended.
"""


def run(cmd, shell=True, check=True, verbose=False, capture_output=True):
    if verbose:
        print(cmd)
    try:
        result = subprocess.run(
            cmd, shell=shell, check=check, capture_output=capture_output, text=True
        )
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        raise


def check_depends(cmd):
    """Takes a failed install command and reruns it, examining the output for dependency issues.
    It collects the names of the interrelated packages and prints them to stdout so the results
    can be used in bash code to correct the failed install."""
    matches = []
    while True:
        try:
            run(cmd)
            break; # In case a non-failing command is given
        except subprocess.CalledProcessError as e:
            output = e.stdout + e.stderr
            new_matches = []
            for line in output.splitlines():
                match = re.match(r"^\s*([^\s]*).*(?:Depends|Breaks): ([^\s]*)", line)
                if match:
                    for g in match.groups():
                        new_matches.append(g)
            if new_matches:
                matches.extend(new_matches)
                cmd += f" {' '.join(new_matches)}"
            else:
                break
    print(" ".join(matches))


__VAR_DICT = {}


def required_key(key):
    return f"Missing required key: {key}"


def required_value(key):
    return f"Missing value for key: {key}"


def check_key(key, envs, parent="", check_value=True):
    keystr = f"{parent}:{key}" if parent else key
    if not key in envs:
        print(required_key(keystr))
        return False
    elif check_value and not envs[key]:
        print(required_value(keystr))
        return False
    return True


def replace_vars(string: str, val_dict: Dict[str, str], permissive: bool) -> str:
    """Replace any variable defined in the vars section

    Args:
        string (str): The string to replace the variables in
        val_dict (dict[str, str]): The dictionary of variables
        permissive (bool): If true, ignore variables that are not defined

    Returns:
        str -- The string with the variables replaced

    Throws:
        RuntimeError -- If the variable is not defined
    """
    pattern = r'\$\{[^}]*\}'
    instances = []
    while True:
        instances = re.findall(pattern, string)
        if not instances:
            break
        variables = {instance[2:-1]: False for instance in instances}
        for var in variables.keys():
            if var not in val_dict:
                if permissive:
                    continue
                raise RuntimeError(f"Variable {var} is not defined")
            variables[var] = True
            string = string.replace("${" + var + "}", val_dict[var])
        # If we are permissive, we can stop if not all variables are defined
        if permissive and not all(variables.values()):
            break
    return string


def check_vars(key, envs):
    """Check if we have defined any configuration variables"""
    # the vars key is optional and if not defined it can be omitted
    if "vars" not in envs:
        return True
    global __VAR_DICT
    __VAR_DICT = envs['vars'].copy()
    # validate that the variables is a Dict[str, str]
    for k, v in __VAR_DICT.items():
        if not isinstance(k, str):
            raise RuntimeError(f"Variable {k} is not a string")
        if not isinstance(v, str):
            raise RuntimeError(f"Value {v} is not a string")
    # Overwrite the values of the variables from the environment if they exist
    for k, _ in __VAR_DICT.items():
        if k in os.environ:
            __VAR_DICT[k] = os.environ[k]
    # Substitute any variables that reference other variables
    for k, v in __VAR_DICT.items():
        # Non permissive variable substitution: Do not allow any undefined variables in
        # the vars section. Raise an error if we reference a variable that is not defined
        # in the definition of another variable. We expect that we can
        # fully expand the values of the defined variable at parsing time.
        res = replace_vars(v, __VAR_DICT, permissive=False)
        if res:
            __VAR_DICT[k] = res
    envs['vars'] = __VAR_DICT
    return True


def check_os(envs):
    ok = check_key('os', envs)
    if check_key('name', envs['os'], 'os'):
        if not envs['os']['name'] == "Ubuntu":
            print("os:name Installer supports Ubuntu only")
            ok = False
    else:
        ok = False
    ok = ok and check_key('version', envs['os'], 'os')
    return ok


def check_system_libs(key, libs):
    # system libs are always optional
    if key in libs:
        if libs[key]:
            return True
        else:
            print(required_value(key))
            return False
    else:
        return True


def check_is_list(value, text, of_dict: bool = False):
    if isinstance(value, list) and (not of_dict or all(isinstance(item, MutableMapping) for item in value)):
        return True
    else:
        extra = " of dicts" if of_dict else ""
        print(f"{text} must be a (possibly empty) list{extra}")
        return False


def check_subset_or_component(pkg: str, envs, component: bool = False, required: bool=True):
    if required:
        if not check_key(pkg, envs):
            return False
    elif not pkg in envs:
        return True
    
    prefix = 'penv:' if not component else ''

    if (component or check_key('index_url', envs[pkg], pkg)) and \
        check_key('libs', envs[pkg], pkg, check_value=False):
        libs = envs[pkg]['libs']
        optionals = envs[pkg].get('optionals', [])
        return check_is_list(libs, f"{prefix}{pkg}:libs") and \
            check_is_list(optionals, f"{prefix}{pkg}:optionals (if present)", of_dict=True)


def check_penv(envs):
    # penv is optional
    ok = True
    if not 'penv' in envs:
        return True
    if not check_key('penv', envs):
        return False
    envs = envs['penv']
    if check_key('python', envs):
        if not str(envs['python']).startswith("3."):
            print("penv:python: Installer supports python3 only")
            ok = False
    else:
        ok = False
        ok = ok and check_key('pip', envs)

    ok = ok and check_key('repositories', envs)

    ok = ok and check_subset_or_component('common', envs)
    ok = ok and check_subset_or_component('axelera_common', envs)
    ok = ok and check_subset_or_component('axelera_development', envs)
    ok = ok and check_subset_or_component('axelera_runtime', envs)
    ok = ok and check_subset_or_component('development', envs)
    ok = ok and check_subset_or_component('runtime', envs)
    ok = ok and check_subset_or_component('torch', envs)
    ok = ok and check_subset_or_component('llm', envs, required=False)
    
    ok = ok and check_key('requirements', envs)
    ok = ok and check_key('pyenv', envs)
    ok = ok and check_key('python_src', envs)
    ok = ok and check_key('python_dependencies', envs)
    ok = ok and check_key('pyenv_dependencies', envs)
    return ok


def validate(envs):
    ok = True
    ok = ok and check_key('name', envs)
    ok = ok and check_vars('vars', envs)
    ok = ok and check_os(envs)
    ok = ok and check_system_libs('installer_dependencies', envs)
    ok = ok and check_penv(envs)

    ok = ok and check_subset_or_component('docker', envs, component=True)
    ok = ok and check_subset_or_component('runtime', envs, component=True)
    ok = ok and check_subset_or_component('common', envs, component=True)
    ok = ok and check_subset_or_component('driver', envs, component=True)
    return ok


def limit_for_hash(env, hash_type, optional):
    # Remove all settings that don't affect docker contents
    is_docker_hash = hash_type == "docker"
    if 'penv' in env:
        if 'repositories' in env['penv']:
            del env['penv']['repositories']
        if 'pyenv' in env['penv']:
            del env['penv']['pyenv']
        if 'python_src' in env['penv']:
            del env['penv']['python_src']
        if 'pyenv_dependencies' in env['penv']:
            del env['penv']['pyenv_dependencies']
    for component in ['docker', 'runtime', 'common']:
        if component in env:
            if not is_docker_hash:
                del env[component]
            else:
                if 'description' in env[component]:
                    del env[component]['description']
                if 'system-dependencies' in env[component]:
                    del env[component]['system-dependencies']
                if 'optionals' in env[component] and not optional:
                    del env[component]['optionals']
    if 'driver' in env:
        del env['driver']
    if 'repositories' in env:
        del env['repositories']
    if 'media' in env:
        del env['media']
    if 'next' in env:
        del env['next']


def flatten(env, parent=False, optional=False):
    items = []
    optionals = env.pop('optionals', [])
    for key, value in env.items():
        key = str(parent) + str("_") + str(key) if parent else key
        if isinstance(value, MutableMapping):
            _items, _unused = flatten(value, key, optional=optional)
            items.extend(_items.items())
            optionals += _unused
        elif isinstance(value, list):
            if optional and optionals and key.endswith('_libs'):
                for item in optionals:
                    value.append(item['name'])
                optionals = []
            for k, v in enumerate(value):
                _items, _unused = flatten({str(k): v}, key, optional=optional)
                items.extend(_items.items())
                optionals += _unused
        else:
            items.append((key, value))
    return dict(items), optionals


def parse_config_file(args):
    envs = {}

    try:
        with open(args.config_file, "r") as f:
            envs = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)
        sys.exit(1)

    if not validate(envs):
        sys.exit(1)

    if args.hash:
        limit_for_hash(envs, args.hash, args.optional)
        yaml_str = json.dumps(envs, sort_keys=True)
        with open(args.requirements_file, 'r') as file:
            yaml_str += file.read()
        hash_str = hashlib.sha256(yaml_str.encode('utf-8')).hexdigest()
        print(hash_str[:8])
    else:
        flat, unused_optionals = flatten(envs, optional=args.optional)
        for ix, unused in enumerate(unused_optionals):
            key = f'unused_optionals_{ix}'
            flat[f'{key}_name'] = unused['name']
            flat[f'{key}_info'] = unused['info']

        for i in flat.items():
            # Substitute the values of any variables
            key = i[0].replace('-', '_').replace('.', '_')
            # Variable substitution is permissive in the body of the configuration
            # If there are undefined variables then we expect that those are
            # system environment variables that will be defined at runtime.
            value = replace_vars(str(i[1]), __VAR_DICT, permissive=True)
            value=str(value).replace('"', '\\"').replace('$', '\\\\\\$')
            print(f'export AX_{key}="{value}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Support functions for the installer.")
    top_level = parser.add_mutually_exclusive_group()

    top_level.add_argument("--check-depends", type=str, help="run `check_depends` on the given command")

    top_level.add_argument("--config-file", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--hash", type=str, choices=["docker", "env"], help="Type of hash to compute")
    parser.add_argument("--requirements-file", type=str, help="Path to the requirements file")
    parser.add_argument("--optional", action='store_true', help="Include optional packages")

    args = parser.parse_args()

    if (args.hash is None) != (args.requirements_file is None):
        parser.error("Either both --hash and --requirements-file must be provided, or neither.")

    if args.check_depends:
        check_depends(args.check_depends)
    elif args.config_file:
        parse_config_file(args)
    else:
        parser.print_help()
        exit(1)
